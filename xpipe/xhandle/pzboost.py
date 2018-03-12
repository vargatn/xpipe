"""
Source pdf-container and boost factor creation

The philosophy for decomposition is that it is done for each JK patch independently...
"""

import os
import scipy.optimize as optimize
import numpy as np
import fitsio as fio
import pickle
import multiprocessing as mp
import pandas as pd

from ..tools import selector as sl
from .. import paths

pcols = [('lens_id', 'i8'), ('source_id', 'i8'), ('rbin', 'i4'), ('source_weight', 'f8'), ('scinv', 'f8'),
         ('gsens_t', 'f8'), ('z_sample', 'f8')]
patch_id = [('patch_id', 'i4'),]

BADVAL = -9999

raw_pdf_tag = "_pzcont_"
# raw_pdf_tag = "_resp_pzcont_"
pwsum_suffix = "_pwsum.npz"
# pwsum_suffix = "_resp_pwsum.npz"



# TODO document this ASAP!!!
"""creates infodicts for extracting PDFs"""


###################################################################
# constructing stacked PDFs based on source-lens pair logs

def create_infodicts(pairs_names, bin_vals=None, npatch=None, pdf_paths=None, force_zcens=None,
                     pdfid='coadd_objects_id', source_id="source_id", force_rbin=None,
                     histpars=None, fullpars=None, boostpars=None):
    """
    Creates infodicts for extracting PDFs

    Parameters
    ----------
    pairs_names : list
        full path of JK pairs files
    bin_vals : array
        array containing bin IDs
    npatch : int
        number of JK regions
    pdf_paths : list
        full paths of P(z) PDFs
    force_zcens : array
        redshift centers to enfore
    pdfid : str
        key or column in the PDF table
    source_id : str
        key or column in the source table
    force_rbin : int
        index of radial bin to enforce
    histpars : dict
        p(z) histogram parameters
    fullpars : dict
        p(z) PDF parameters
    boostpars : dict
        boost decomposition parameters

    Returns
    -------
    list
        list of infodicts

    Notes
    -----
    Structure of the dictionary::

            infodict = {
                'npatch': npatch,
                'pname':pname,
                'bin_val': bin_vals[i],
                'pdf_paths': pdf_paths,
                'force_zcens': force_zcens,
                'pdfid': pdfid,
                'source_id': source_id,
                'force_rbin': force_rbin,
                'histpars': histpars,
                'fullpars:': fullpars,
                'boostpars': boostpars,
            }

    """
    master_infodicts = []
    for i, clust_pnames in enumerate(pairs_names):
        infodicts = []
        for j, pname in enumerate(clust_pnames):
            infodict = {
                'npatch': npatch,
                'pname':pname,
                'bin_val': bin_vals[i],
                'pdf_paths': pdf_paths,
                'force_zcens': force_zcens,
                'pdfid': pdfid,
                'source_id': source_id,
                'force_rbin': force_rbin,
                'histpars': histpars,
                'fullpars:': fullpars,
                'boostpars': boostpars,
            }
            infodicts.append(infodict)
        master_infodicts.append(infodicts)
    return master_infodicts


def multi_pwsum_run(infodicts, nprocess=1):
    """
    OpenMP style parallelization for tasks

    Separates tasks into chunks, and passes each chunk for an independent process

    Parameters
    ----------
    infodicts : list
        infodicts which list instructions on what to do
    nprocess : int
        number of OpenMP style parallelization processes. This is the maximum value, the actual is
        max(len(infodicts), nprocess)
    """
    # at most as many processes can be used as there are independent tasks...
    if nprocess > len(infodicts):
        nprocess = len(infodicts)

    print 'starting PDF extraction in ' + str(nprocess) + ' processes'
    fparchunks = sl.partition(infodicts, nprocess)
    pool = mp.Pool(processes=nprocess)
    pp = pool.map_async(call_pwsum_chunk, fparchunks)

    try:
        pp.get(86400)  # apparently this counters a bug in the exception passing in python.subprocess...
    except KeyboardInterrupt:
        print "Caught KeyboardInterrupt, terminating workers"
        pool.terminate()
        pool.join()
    else:
        pool.close()
        pool.join()


def balance_infodicts(master_infodicts, ibin, nchunk, ichunk):
    """
    Distribute infodicts across nodes

    Parameters
    ----------
    master_infodicts : list
        infodicts which list instructions on what to do
    ibin : int
        index of parameter bin, default is None
    nchunk : int
        how many chunks to create
    ichunk : int
        which chunk to return

    Returns
    -------
    list
        infodicts selected from the master
    """
    raw_infodicts = master_infodicts
    if ibin is not None:
        raw_infodicts = master_infodicts[ibin]

    infodicts = np.array(raw_infodicts).flatten()
    if nchunk > 1:
        tmp_infodicts = partition_tasks(raw_infodicts, nchunk)
        infodicts = tmp_infodicts[ichunk]

    return infodicts


def partition_tasks(master_infodicts, tasks):
    """split LARGE job into chunks"""
    flat_master = np.array(master_infodicts).flatten()
    _tasks = np.min((len(flat_master), tasks)).astype(int)
    fparchunks = sl.partition(flat_master, _tasks)
    return fparchunks


def call_pwsum_chunk(indodicts):
    """Executes a single chunk serially (simple for loop"""
    for infodict in indodicts:
        extract_pwsum(infodict)


def extract_pwsum(infodict):
    """Extracts w * PDF from hdf5 files"""
    fname = infodict['pname']
    pdf_paths = infodict['pdf_paths']
    oname = fname.replace('.dat', pwsum_suffix)
    fbase = os.path.split(fname)[1].split('_result')[0]

    force_rbin = infodict['force_rbin']
    if os.path.isfile(fname):
        print "starting", fname

        raw_data = np.loadtxt(fname, dtype=pcols)
        scat = pd.DataFrame.from_records(raw_data)

        # identify rbins
        rvals = scat['rbin'].values
        rbvals = np.unique(rvals)

        pwsums = []
        wsums = []
        rbins = []
        nobjs = []
        # select rbin subsets
        for rbval in rbvals:
            if force_rbin is None or rbval == force_rbin:
                print fbase + ' rbin ' + str(rbval)
                rsub = scat.query('rbin == ' + str(rbval))[:5000]
                pwsum, wsum, zcens, nobj = calc_pwsum(rsub, pdf_paths, pdfid=infodict['pdfid'],
                                                          source_id=infodict['source_id'],
                                                          force_zcens=infodict['force_zcens'])
                pwsums.append(pwsum)
                wsums.append(wsum)
                rbins.append(rbval)
                nobjs.append(nobj)

        if len(rbins):
            np.savez(oname, pwsums=pwsums, wsums=wsums, rbvals=rbvals,
                     zcens=zcens, nobjs=nobjs)
            print 'written ' + oname
    else:
        print "no such file:", fname


def calc_pwsum(pscat, pdf_paths, pdfid='INDEX', source_id="source_id", force_zcens=None):
    """Calculates w * PDF and w for a single JK-patch. Writes directly to disk"""
    zcens = force_zcens
    if force_zcens is None:
        with pd.HDFStore(pdf_paths[0], 'r') as hh:
            zcens = hh['info'].values[:, 0]
    pwsum = np.zeros(len(zcens))
    wsum = 0.0
    n_obj = 0
    for ppath in pdf_paths:
        with pd.HDFStore(ppath, 'r') as pp:
            # match IDs from pdf storage with sources
            zvals = pp.select(key='point_predictions', columns=(pdfid,)).reset_index(drop=True)
            zvals['rowind'] = zvals.index.values
            left_match = pscat.merge(zvals, how='inner',
                                     left_on=source_id, right_on=pdfid).dropna()
            rowind = left_match['rowind'].values.astype(int)

            if len(rowind):
                # loading rows from PDF container
                cols_to_use = pp.select('pdf_predictions', where=(0,)).keys()[4:].values
                pdf_rows = pp.select(key='pdf_predictions', where=rowind, columns=cols_to_use).values
                # assigning weights
                ww = left_match['source_weight'].values * left_match["scinv"].values * left_match["gsens_t"]

                #saving values
                pwsum += np.sum(pdf_rows * ww[:, np.newaxis], axis=0)
                wsum += np.sum(ww)
                n_obj += len(ww)

    return pwsum, wsum, zcens, n_obj


def check_pwsum_files(pnames, npatch):
    """Tests wether the JK patch has a PDF extracted and saved"""
    # FIXME this might have some path inconsistencies

    has_pairsfile = np.zeros(npatch, dtype=bool)
    fnames = []
    for pid, pname in enumerate(pnames):
        fname = paths.dirpaths['xout'] + '/' + pname.replace('.dat', pwsum_suffix)
        fnames.append(fname)
        if os.path.isfile(fname):
            has_pairsfile[pid] = True
    return fnames, has_pairsfile


def _check_zcens(pnames, has_pairsfile):
    """reads z-grid centers from pdf_chunk"""
    fname = pnames[np.nonzero(has_pairsfile)[0][0]]
    zdata = dict(np.load(fname))
    zcens = zdata['zcens']
    return zcens


def combine_pwsums(infodict):
    """Combines pwsums from JK patches into a PDFContainer object"""
    # TODO check file path consistency
    npatch = infodict[0]['npatch']
    pdf_paths = infodict[0]['pdf_paths']
    pnames = [infod['pname'] for infod in infodict]
    bin_vals = np.array([infod['bin_val'] for infod in infodict])

    if not (bin_vals - bin_vals[0] == 0).all():
        raise ValueError("the JK patches are mixed up, aborting...")

    fullpars = paths.params["pzpars"]["full"]
    if "fullpars" in infodict[0].keys() and infodict[0]["fullpars"] is not None:
        fullpars.update(infodict[0]["fullpars"])

    fnames, has_pairsfile = check_pwsum_files(pnames, npatch)
    # print fnames
    zcens = _check_zcens(fnames, has_pairsfile)
    nrbins = paths.params['radial_bins']['nbin']
    nzbins = len(zcens)
    pdf_subs = np.zeros(shape=(nrbins, npatch, nzbins))
    w_subs = np.zeros(shape=(nrbins, npatch))
    nsources = np.zeros(shape=(nrbins, npatch), dtype=int)

    for pid in np.arange(npatch):
        if has_pairsfile[pid]:
            print 'lbin:',bin_vals[pid][0], 'zbin:', bin_vals[pid][1],'processing patch', pid
            for i, fname in enumerate(fnames):
                if has_pairsfile[i] and i != pid:
                    zdata = dict(np.load(fname))
                    for ri, rb in enumerate(zdata['rbvals']):
                        pdf_subs[rb, pid] += zdata['pwsums'][ri]
                        w_subs[rb, pid] += zdata['wsums'][ri]
                        nsources[rb, pid] += zdata['nobjs'][ri]
            pdf_subs[:, pid] /= w_subs[:, pid][:, np.newaxis]

    pcont = PDFContainer(oname=None, haspairs=has_pairsfile, npatch=npatch,
                         nrbin=nrbins, zcens=zcens, pdf_paths=pdf_paths)

    pcont.nsources = nsources
    pcont.pdf_subs = pdf_subs
    pcont.normsub()

    # TODO this function no longer exists
    corename = paths.get_corename(pnames[0])
    bdict = pcont.to_dict()

    bname = paths.dirpaths['xout'] + '/' + corename + raw_pdf_tag + fullpars['tag'] + '.p'
    print bname
    pickle.dump(bdict, open(bname, 'wb'))


###################################################################
# Container for Boost Factor PDFsubpatches and decomposition

class PDFContainer(object):
    def __init__(self, haspairs, npatch, nrbin, oname=None,
                 zcens=None, zmin=None, zmax=None, nzbin=None, **kwargs):
        self.oname = oname
        self.haspairs = haspairs.astype(bool)
        self.npatch = npatch
        self.pindexes = np.arange(self.npatch)[self.haspairs]
        self.nrbin = nrbin
        self.rbins = np.arange(self.nrbin)
        self.rcens = paths.cens[self.rbins]

        self.pexts = None
        # FITS data file
        if self.oname is not None:
            self.pfits = fio.FITS(self.oname)
            self.pexts = self._get_patch_exts()
        # print "5"
        # pdf binning parameters
        if zcens is not None and zmin is None and zmax is None and nzbin is None:
            self.zcens = zcens
            self.zmin, self.zmax, self.zedges = self._get_zedges()
            self.nbin = len(self.zcens)

        elif zcens is None and zmin is not None and zmax is not None and nzbin is not None:
            self.zmin = zmin
            self.zmax = zmax
            self.nbin = nzbin
            self.zedges, self.zcens = self._get_zarr()
        else:
            raise TypeError
        # print "6"

        # pdf averaging parameters
        self.rkey = 'rbin'
        self.wkey = 'source_weight'
        self.zkey = 'z_sample'

        # Data containers
        self.nsources = np.zeros(shape=(self.nrbin, self.npatch))
        self.pdf_subs = np.zeros(shape=(self.nrbin, self.npatch, len(self.zcens)))
        self.pdfs = None
        self.pdf_errs = None

        # WARNING this is really just the values in each patch, not the usual JK-notation (i.e except-th patch)
        self.pwsum_subs = np.zeros(shape=(self.nrbin, self.npatch, len(self.zcens)))
        self.wsum_subs = np.zeros(shape=(self.nrbin, self.npatch))

        # Additional calculated values
        self.bw = np.mean(np.diff(self.zedges))

        # boost factor initial values
        self.mean_init = 0.5
        self.sigma_init = 0.1
        self.amp_init = 0.5

        self.mean = BADVAL
        self.mean_err = BADVAL
        self.sigma = BADVAL
        self.sigma_err = BADVAL
        self.amps = None
        self.amps_err = None
        self.amps_cov = None

        self.boost_rvals = None
        self.boost = None
        self.boost_err = None
        self.boost_cov = None

        self.point = None
        self.point_subs = None
        self.point_cov = None
        self.goodinds = None

        # Boost factor bounds
        self.mean_bounds = (0., np.inf)
        self.sigma_bounds = (0., np.inf)
        self.amp_bounds_single = (0., 1.)

        self.verbose_prefix = ''

    def to_dict(self):
        infodict = {
            'oname': self.oname,
            'haspairs': self.haspairs,
            'npatch': self.npatch,
            'pindexes': self.pindexes,
            'nrbin': self.nrbin,
            'pexts': self.pexts,
            'zmin': self.zmin,
            'zmax': self.zmax,
            'nbin': self.nbin,
            'rkey': self.rkey,
            'wkey': self.wkey,
            'zkey': self.zkey,
            'nsources': self.nsources,
            'pdf_subs': self.pdf_subs,
        }
        return infodict

    @classmethod
    def from_dict(cls, infodict):

        pdfcont = cls(oname=infodict['oname'], haspairs=infodict['haspairs'],
                      zmin=infodict['zmin'], zmax=infodict['zmax'],
                      nzbin=infodict['nbin'], npatch=infodict['npatch'], nrbin=infodict['nrbin'])

        # pdf averaging parameters
        pdfcont.rkey = infodict['rkey']
        pdfcont.wkey = infodict['wkey']
        pdfcont.zkey = infodict['zkey']

        # Data containers
        pdfcont.nsources = infodict['nsources']
        pdfcont.pdf_subs = infodict['pdf_subs']

        return pdfcont

    @classmethod
    def from_file(cls, fname):
        """Loads self from pickled dict"""
        infodict = pickle.load(open(fname, 'rb'))
        pdfcont = cls.from_dict(infodict)
        return pdfcont

    def to_boostdict(self):
        """saving Boost results"""

        boostdict = {
            'rbins': self.rbins,
            'rvals': self.rcens,
            'amps': self.amps,
            'amps_err': self.amps_err,
            'mean': self.mean,
            'mean_err': self.mean_err,
            'sigma': self.sigma,
            'sigma_err': self.sigma_err,
            'boost_rvals': self.boost_rvals,
            'boost': self.boost,
            'boost_err': self.boost_err,
            'boost_cov': self.boost_cov,
        }
        return boostdict

    def _get_patch_exts(self):
        """Creates string id for patches"""
        pexts = []
        for i in self.pindexes:
            pexts.append('patch_' + str(i))
        return pexts

    def _get_zarr(self):
        """get redshift nbin edges"""
        zedges = np.linspace(self.zmin, self.zmax, self.nbin + 1)
        zcens = np.diff(zedges) / 2. + zedges[:-1]
        return zedges, zcens

    def _get_zedges(self):
        """get redshift grid params from centers"""
        zdiff = np.mean(np.diff(self.zcens))
        zedges = np.concatenate((self.zcens - zdiff / 2., [self.zcens[-1] + zdiff / 2., ]))
        zmin = zedges[0]
        zmax = zedges[-1]
        return zmin, zmax, zedges

    def get_nsources(self, rb):
        """Checks which patch has how many sources for which bin"""
        nsources = np.zeros(self.npatch, dtype=int)
        for pext, lab in zip(self.pexts, self.pindexes):
            # loading HDUtable for JK-patch
            dtable = self.pfits[pext]

            rbins = dtable[self.rkey][:]
            rinds = np.where(rbins == rb)[0]
            nsources[lab] = len(rinds)
        self.nsources[rb] = nsources

    def _calc_histogram(self, zvals, weights):
        """histogram """
        counts, tmp = np.histogram(zvals, self.zedges, weights=weights)
        return counts

    def _calc_pdf_patch_hist(self, rbin, pid):
        """
        Loops trough all JK-patches and adds up the z-histogram of the relevant sources
        """

        # looping trough all JK-patches except the selected id, and
        psub = np.zeros(len(self.zcens))
        for pext, lab in zip(self.pexts, self.pindexes):
            if lab != pid and self.nsources[rbin][lab]:
                # loading HDUtable for JK-patch
                dtable = self.pfits[pext]

                # selecting pairs in radial bin
                rbins = dtable[self.rkey][:]
                rinds = np.where(rbins == rbin)[0]

                # getting redshit value and weights
                zvals = dtable[self.zkey][rinds]
                weights = dtable[self.wkey][rinds]
                counts = self._calc_histogram(zvals, weights)
                psub += counts

        # norming the histogram
        psub = self._norm_pdf(psub)
        return psub

    def _norm_pdf(self, pdf):
        norm = np.sum(self.bw * pdf)
        return pdf / norm

    def normsub(self):
        for rb in np.arange(self.nrbin):
            for jpatch in np.arange(self.npatch)[self.haspairs]:
                self.pdf_subs[rb, jpatch] = self._norm_pdf(self.pdf_subs[rb, jpatch])

    def _get_pdf_patch(self, rbin, pid):
        """Returns the pdf for a single JK-patch"""
        psub = np.zeros(len(self.zcens))
        # making sure that this JK-patch *has* sources for this radial bin
        if self.nsources[rbin][pid]:
            print self.verbose_prefix + ' rbin: ' + str(rbin) + ' patch: ' + str(pid)
            psub = self._calc_pdf_patch_hist(rbin, pid)
        return psub

    def _get_pdf_rbin(self, rbin, verbose_prefix=''):
        """Creates the pdf-s for each patch"""
        self.get_nsources(rbin)
        for pid in np.arange(self.npatch):
            self.pdf_subs[rbin][pid] = self._get_pdf_patch(rbin, pid)

    def get_hist_pdfs(self, verbose_prefix=''):
        """Calculates pdf_sub for each rbin"""
        self.verbose_prefix = verbose_prefix
        for rb in np.arange(self.nrbin):
            self._get_pdf_rbin(rb, verbose_prefix)

    def _get_mean_pdf(self, rbin):
        """Calculates the JK estimate on the pdf"""
        ipatches = np.nonzero(self.nsources[rbin])[0]
        njk = len(np.nonzero(self.nsources[rbin])[0])

        pdf = np.zeros(self.nbin)
        pdf_err = np.zeros(self.nbin)
        if njk > 1:
            pdf = np.sum(self.pdf_subs[rbin, ipatches], axis=0) / njk
            pdf_err = np.sqrt(np.sum((self.pdf_subs[rbin, ipatches] - pdf)**2, axis=0) * (njk - 1.0) / njk )
        return pdf, pdf_err

    def get_mean_pdfs(self):
        """Calculates the JK estimate on the pdf and its's errors"""
        self.pdfs = np.zeros(shape=(self.nrbin, self.nbin))
        self.pdf_errs = np.zeros(shape=(self.nrbin, self.nbin))
        for rb in np.arange(self.nrbin):
            self.pdfs[rb], self.pdf_errs[rb] = self._get_mean_pdf(rb)

    def _decomp_gauss_patch(self, refprof, ipatch=2, rbins=(5, 6)):
        """Decomposes to a mixture of gaussian and reference profile"""
        npdf = len(rbins)
        # fit parameter limits for least squares
        bounds = np.array([self.mean_bounds, self.sigma_bounds] + npdf * [self.amp_bounds_single,]).T
        # initial values for least squares
        point_init = np.array([self.mean_init, self.sigma_init] + npdf * [self.amp_init,])

        pdf = self.pdf_subs[rbins, ipatch, :]
        if isinstance(refprof, PDFContainer):
            refpdf = refprof.pdf_subs[rbins, ipatch, :]
            bmixer = BoostMixerRandRef(self.zcens, pdf, refpdf)
        else:
            # pdf values to use
            refpdf = self.pdf_subs[refprof, ipatch, :]
            bmixer = BoostMixer(self.zcens, pdf, refpdf)

        res = optimize.least_squares(bmixer, point_init, bounds=bounds)
        return res

    def _decomp_gauss_jk(self, refprof, force_rbmin=4, force_rbmax=9):
        """Calculates the boost factor parameters for each jk-patch"""

        self.point_subs = np.ones(shape=(2 + self.nrbin, self.npatch)) * BADVAL

        # setting proper radial bin limits
        self.rbins = np.arange(self.nrbin)
        rbmin = self.rbins.min()
        rbmax = self.rbins.max()
        if force_rbmin is not None:
            rbmin = force_rbmin
        if force_rbmax is not None:
            rbmax = force_rbmax

        for ipatch in np.arange(self.npatch):
            sbins = np.nonzero(self.nsources[:, ipatch])[0]

            # these are the radial bin indices which we actually want to use in the Boost factor estimation
            rbins_to_use = np.sort(list(set(sbins).intersection(set(self.rbins))))
            rbins_to_use = rbins_to_use[np.where((rbins_to_use >= rbmin) * (rbins_to_use <= rbmax))[0]]

            if len(rbins_to_use) != 0 and np.min(rbins_to_use) > rbmin:
                rbins_to_use = []

            # print sbins.shape, sbins
            if len(rbins_to_use) and (refprof in sbins or isinstance(refprof, PDFContainer)):
                res = self._decomp_gauss_patch(refprof, ipatch, rbins_to_use)
                if res['success']:
                    self.point_subs[:2, ipatch] = res['x'][:2]
                    self.point_subs[2 + rbins_to_use, ipatch] = res['x'][2:]

    def decomp_gauss(self, refprof, force_rbmin=None, force_rbmax=None):
        """Estimates boost factors from P(z) decomposition"""
        self._decomp_gauss_jk(refprof, force_rbmin, force_rbmax)
        self._get_boost_err()

    def _get_boost_err(self):
        self.point = np.ones(2 + self.nrbin) * BADVAL
        self.point_err = np.ones(2 + self.nrbin) * BADVAL
        self.point_cov = np.ones((2 + self.nrbin, 2 + self.nrbin)) * BADVAL

        # checking which values can we use to get an error estimate (
        good_patches = np.zeros(self.point_subs.shape, dtype=bool)
        for i, pars in enumerate(self.point_subs):
            for j, patch in enumerate(pars):
                if self.point_subs[i, j] != BADVAL:
                    good_patches[i, j] = True

        # calculating means for each parameter
        for i, (pp, guids) in enumerate(zip(self.point_subs, good_patches)):
            njk = sum(guids)
            if njk > 1:
                self.point[i] = np.sum(self.point_subs[i, guids]) / njk

        # calculating covariance for each parameter
        for i, (pp1, guids1) in enumerate(zip(self.point_subs, good_patches)):
            for j, (pp2, guids2) in enumerate(zip(self.point_subs, good_patches)):
                guids = guids1 * guids2
                njk = np.sum(guids)

                if njk > 1:
                    self.point_cov[i, j] = (np.sum((self.point_subs[i, guids] - self.point[i, np.newaxis]) *
                                                   (self.point_subs[j, guids] - self.point[j, np.newaxis])) *
                                            (njk - 1.0) / njk)

        self.goodinds = good_patches.sum(axis=1).astype(bool)
        self.point_err[self.goodinds] = np.sqrt(np.diag(self.point_cov)[self.goodinds])
        self.mean = self.point[0]
        self.mean_err = self.point_err[0]
        self.sigma = self.point[1]
        self.sigma_err = self.point_err[1]
        self.amps = self.point[2:]
        self.amps_err = self.point_err[2:]
        self.amps_cov = self.point_cov[2:, 2:]

        hasamp = np.where(self.amps != BADVAL)[0]
        self.boost_rvals = self.rcens[hasamp]
        self.boost = 1. + self.amps[hasamp]
        self.boost_err = self.amps_err[hasamp]
        self.boost_cov = self.amps_cov[hasamp, :][:, hasamp]


def get_hist_zarr():
    zmin = paths.params['pzpars']['zmin']
    zmax = paths.params['pzpars']['zmax']
    nbin = paths.params['pzpars']['nbin']
    zedges = np.linspace(zmin, zmax, nbin + 1)
    zcens = np.diff(zedges) / 2. + zedges[:-1]
    return zcens, zedges


###################################################################
# Boost factor simple Gaussian decomposition

def gauss(xx, m=0.0, s=1.0):
    """
    1D Gaussian

    :param x: data array

    :param m: mean
    :param s: sigma
    :returns: y
    """

    anorm = 1. / (np.sqrt(2. * np.pi) * s)
    yy = anorm * np.exp(-(1. / 2.) * (xx - m) ** 2. / s ** 2)
    return yy


class BoostMixer(object):
    def __init__(self, zcens, pdfarr, refpdf):
        """
        Decomposes PDF into a gaussian + reference

        the mean, width of the gaussian is fixed for all pdf in pdfarr, the mixing amplitude is free

        :param zcens: centers of redshift bins
        :param pdfarr: list of pdfs to decompose (e.g. radial bins
        :param refpdf: single reference pdf to use in the decomposition
        """

        self.zcens = zcens
        self.bw = np.mean(np.diff(self.zcens))

        self.pdfarr = np.array([self._norm(pdf) for pdf in pdfarr])
        self.refpdf = self._norm(refpdf)

        self.mixarr = np.zeros(shape=self.pdfarr.shape)
        self.garr = np.zeros(shape=self.refpdf.shape)

    def pgauss(self, m=0.0, s=1.0):
        """Gaussian normalized to redshift grid"""
        garr = gauss(self.zcens, m=m, s=s)
        return self._norm(garr)

    def _norm(self, pdf):
        """Normalizes pdf to the current redshift grid"""
        norm = np.sum(self.bw * pdf)
        return pdf / norm

    def mixer(self, mean, sigma, amps):
        """Creates mixture of reference and gaussian pdf-s"""
        garr = self.pgauss(m=mean, s=sigma)
        for i, aa in enumerate(amps):
            self.mixarr[i, :] = amps[i] * garr + (1. - amps[i]) * self.refpdf

    def __call__(self, point):
        """Calculate residual"""
        mean = point[0]
        sigma = point[1]
        amps = point[2:]

        self.mixer(mean, sigma, amps)

        diff = self.mixarr - self.pdfarr
        return diff.flatten()


class BoostMixerRandRef(object):
    def __init__(self, zcens, pdfarr, refpdfarr):
        """
        Decomposes PDFs into a gaussian + reference, in this implementation there is a different reference for each PDF

        the mean, width of the gaussian is fixed for all pdf in pdfarr, the mixing amplitude is free.

        :param zcens: centers of redshift bins
        :param pdfarr: list of pdfs to decompose (e.g. radial bins
        :param refpdf: single reference pdf to use in the decomposition
        """
        self.zcens = zcens
        self.bw = np.mean(np.diff(self.zcens))

        self.pdfarr = np.array([self._norm(pdf) for pdf in pdfarr])
        self.refpdfarr = np.array([self._norm(pdf) for pdf in refpdfarr])

        self.mixarr = np.zeros(shape=self.pdfarr.shape)
        self.garr = np.zeros(shape=self.refpdfarr.shape)

    def pgauss(self, m=0.0, s=1.0):
        """Gaussian normalized to redshift grid"""
        garr = gauss(self.zcens, m=m, s=s)
        return self._norm(garr)

    def _norm(self, pdf):
        """Normalizes pdf to the current redshift grid"""
        norm = np.sum(self.bw * pdf)
        return pdf / norm

    def mixer(self, mean, sigma, amps):
        """Creates mixture of reference and gaussian pdf-s"""
        garr = self.pgauss(m=mean, s=sigma)
        for i, aa in enumerate(amps):
            self.mixarr[i, :] = amps[i] * garr + (1. - amps[i]) * self.refpdfarr[i]

    def __call__(self, point):
        """Calculate residual between"""
        mean = point[0]
        sigma = point[1]
        amps = point[2:]

        self.mixer(mean, sigma, amps)

        diff = self.mixarr - self.pdfarr
        return diff.flatten()

###################################################################