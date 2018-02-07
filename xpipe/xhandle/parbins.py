"""
# TODO add documentation
"""
import kmeans_radec as krd
import numpy as np
import fitsio as fio
import os

from .. import paths
from ..tools.selector import selector, matchdd
from ..tools.catalogs import to_pandas
from .ioshear import makecat


def field_cut(ra, dec, borders):
    """
    Applies RA, DEC cut based on DES field boundaries

    Parameters
    ----------
    ra : np.array
        right ascension
    dec : np.array
        declination
    borders : dict
        dictionary with rectangular boundaries of the selected area

    Notes
    -----

    Define borders like::

        spt = {
            "dec_top" : -30.,
            "dec_bottom" : -60.,
            "ra_left" : 0.,
            "ra_right" : 360.,
        }

    Returns
    -------
    np.array
        bool array indexing the objects within the selected area
    """

    select = np.zeros(len(ra), dtype=bool)

    # so far this is only suitable for the current DES footprint
    if borders['ra_right'] < borders['ra_left']:
        select += ((borders['dec_top'] > dec) * (borders['dec_bottom'] < dec) *
                   ((borders['ra_left'] < ra) + (borders['ra_right'] > ra)))
    else:
        select += ((borders['dec_top'] > dec) * (borders['dec_bottom'] < dec) *
                   (borders['ra_left'] < ra) * (borders['ra_right'] > ra))
    return select


def get_fields_auto(params=None):
    """
    Extracts field boundaries from project :code:`params` dictionary

    Parameters
    ----------
    params : dict
        Pipeline settings in a dictionary format. If :code:`None` then the default
        :py:data:`paths.params` will be used

    Returns
    -------
    dict
        fields to be used with boundaries
    """

    if params is None:
        params = paths.params

    fields = {}
    for name in params["fields_to_use"]:
        fields.update({name : params["fields"][name]})

    return fields


def load_lenscat(params=None, fullpaths=None):
    """
    Loads lens catalog from fits file

    Parameters
    ----------
    params : dict
        Pipeline settings in a dictionary format.
        If :code:`None` then the default :py:data:`paths.params` will be used
    fullpaths : dict
        Pipeline file paths in a dictionary format.
        If :code:`None` then the default :py:data:`paths.fullpaths` will be used

    Returns
    -------
    dict, np.array, record-array
        lens catalog data in format,
        bool array for selection,
        raw data table before selection

    Notes
    ------

    If using custom settings, you have to specify both :code:`params` and :code:`fullpaths`!
    Using inconsitent definitions result in a :code:`SyntaxError`.

    The first output is a dict with the following keys:

        * :code:`id` catalog ID of the lenses

        * :code:`ra` Right Ascension of the lenses

        * :code:`dec` Delination of the lenses

        * :code:`z` Redshift of the lenses

        * :code:`qlist` np.array of quantities with shape (n_lens, n_quantity)
    """

    if params is None and fullpaths is None:
        params = paths.params
        fullpaths = paths.fullpaths
    elif None in (params, fullpaths):
        raise SyntaxError("Some of the arguments are left at default,"
                          " which results in inconsistent behaviour. "
                          "If using custom input, define all arguments!")

    lenscat = fio.read(fullpaths[params["cat_to_use"]]["lens"])
    lenskey = params['lenskey']

    ids = lenscat[lenskey['id']]
    ra = lenscat[lenskey['ra']]
    dec = lenscat[lenskey['dec']]
    z = lenscat[lenskey["z"]]

    select = np.zeros(len(ra), dtype=bool)
    fields = get_fields_auto()
    for name in fields.keys():
        select += field_cut(ra, dec, fields[name])

    # number of parameter columns
    nq = len(lenskey.keys()) - 4
    qlist = np.zeros(shape=(len(ra), nq))
    for ival in np.arange(nq):
        colname = "q" + str(ival)
        qlist[:, ival] = lenscat[lenskey[colname]]

    data = {
        "id" : ids[select],
        "ra" : ra[select],
        "dec" : dec[select],
        "z": z[select],
        "qlist" : qlist[select]
    }

    return data, lenscat[select]


def load_randcat(params=None, fullpaths=None):
    """
    Loads random point catalog from fits file

    Parameters
    ----------
    params : dict
        Pipeline settings in a dictionary format.
        If :code:`None` then the default :py:data:`paths.params` will be used
    fullpaths : dict
        Pipeline file paths in a dictionary format.
        If :code:`None` then the default :py:data:`paths.fullpaths` will be used

    Returns
    -------
    dict, np.array, record-array
        lens catalog data in format,
        bool array for selection,
        raw data table before selection

    Notes
    ------

    If using custom settings, you have to specify both :code:`params` and :code:`fullpaths`!
    Using inconsitent definitions result in a :code:`SyntaxError`.

    The first output is a dict with the following keys:

        * :code:`id` mock ID of the random points

        * :code:`ra` Right Ascension of the random points

        * :code:`dec` Delination of the random points

        * :code:`z` Redshift of the random points

        * :code:`qlist` np.array of mock quantities with shape (n_lens, n_quantity).
          This refers to the mock quantities assigned to the randoms points,

        * :code:`w` weight of the random points
    """

    if params is None and fullpaths is None:
        params = paths.params
        fullpaths = paths.fullpaths
    elif None in (params, fullpaths):
        raise SyntaxError("Some of the arguments are left at default,"
                          " which results in inconsistent behaviour. "
                          "If using custom input, define all arguments!")

    randcat = fio.read(fullpaths[params["cat_to_use"]]["rand"])
    randkey = params['randkey']

    w = randcat[randkey['w']]
    ra = randcat[randkey['ra']]
    dec = randcat[randkey['dec']]
    z = randcat[randkey["z"]]
    ids = np.arange(len(randcat))

    select = np.zeros(len(ra), dtype=bool)
    fields = get_fields_auto()
    for name in fields.keys():
        select += field_cut(ra, dec, fields[name])

    # number of parameter columns
    nq = len(randkey.keys()) - 4
    qlist = np.zeros(shape=(len(ra), nq))
    for ival in np.arange(nq):
        colname = "q" + str(ival)
        qlist[:, ival] = randcat[randkey[colname]]

    data = {
        "id" : ids[select],
        "w" : w[select],
        "ra" : ra[select],
        "dec" : dec[select],
        "z" : z[select],
        "qlist" : qlist[select]
    }

    return data, randcat[select]


def prepare_lenses(bin_settings=None, params=None, fullpaths=None):
    """
    Loads lens data and defines sub-selections for different parameter bins

    Parameters
    ----------
    bin_settings : list
        (parameter_edges, N_randoms)
        can be specified independently from the pipeline settings,
        default is extracted from :code:`params`
    params : dict
        Pipeline settings in a dictionary format.
        If :code:`None` then the default :py:data:`paths.params` will be used
    fullpaths : dict
        Pipeline file paths in a dictionary format.
        If :code:`None` then the default :py:data:`paths.fullpaths` will be used

    Returns
    -------
    data : dict
        Relevant rows and indexes of the lens sample

    Notes
    ------

    If using custom settings, you have to specify both :code:`params` and :code:`fullpaths`!
    Using inconsitent definitions result in a :code:`SyntaxError`.

    The format of the output data table:

        * :code:`id` catalog ID of the lenses

        * :code:`ra` Right Ascension of the lenses

        * :code:`dec` Delination of the lenses

        * :code:`qlist` np.array of quantities with shape (n_lens, n_quantity)

        * :code:`sinds` list of boolean selection indexes for each parameter bin

        * :code:`fullcat` selected rows of the full lens catalog

        * :code:`bounds` for each bin parameter bounds written as :math:`[x_0;\,x_1)`

        * :code:`plpairs` parameter boundaries simple simple list
         (just boundaries, not pairs or tuples)

    """

    if params is None and fullpaths is None:
        params = paths.params
        fullpaths = paths.fullpaths
    elif None in (params, fullpaths):
        raise SyntaxError("Some of the arguments are left at default,"
                          " which results in inconsistent behaviour. "
                          "If using custom input, define all arguments!")

    if bin_settings is None:
        bin_settings = paths.get_bin_settings(params, paths.assign_mode(params))

    data, fullcat = load_lenscat(params, fullpaths)
    sinds, bounds, plpairs = selector(data["qlist"], bin_settings[0])

    data.update({
        "sinds" : sinds,
        "fullcat" : fullcat,
        "bounds" : bounds,
        "plpairs" : plpairs,
    })
    if "jkey" in params["lenskey"].keys() and params["lenskey"]["jkey"] is not None:
        data.update({"jk" : fullcat[params["lenskey"]["jkey"]]})
    return data


def prepare_random(bin_settings=None, params=None, fullpaths=None):
    """
    Loads random points and defines sub-selections for different parameter bins

    Parameters
    ----------
    bin_settings : list
        (parameter_edges, N_randoms)
        can be specified independently from the pipeline settings,
        default is extracted from :code:`params`
    params : dict
        Pipeline settings in a dictionary format.
        If :code:`None` then the default :py:data:`paths.params` will be used
    fullpaths : dict
        Pipeline file paths in a dictionary format.
        If :code:`None` then the default :py:data:`paths.fullpaths` will be used

    Returns
    -------
    data : dict
        Relevant rows and indexes of the lens sample

    Notes
    ------

    If using custom settings, you have to specify both :code:`params` and :code:`fullpaths`!
    Using inconsitent definitions result in a :code:`SyntaxError`.

    The format of the output data table:

        * :code:`id` mock ID of the random points

        * :code:`ra` Right Ascension of the random points

        * :code:`dec` Delination of the random points

        * :code:`qlist` np.array of mock quantities with shape (n_lens, n_quantity).
          This refers to the mock quantities assigned to the randoms points,

        * :code:`w` weight of the random points

        * :code:`sinds` list of boolean selection indexes for each parameter bin

        * :code:`fullcat` selected rows of the full lens catalog

        * :code:`bounds` for each bin parameter bounds written as :math:`[x_0;\,x_1)`

        * :code:`plpairs` parameter boundaries simple simple list
         (just boundaries, not pairs or tuples)
    """

    if params is None and fullpaths is None:
        params = paths.params
        fullpaths = paths.fullpaths
    elif None in (params, fullpaths):
        raise SyntaxError("Some of the arguments are left at default,"
                          " which results in inconsistent behaviour. "
                          "If using custom input, define all arguments!")

    if bin_settings is None:
        bin_settings = paths.get_bin_settings(params, paths.assign_mode(params))

    data, fullcat = load_randcat(params, fullpaths)
    sinds, bounds, plpairs = selector(data["qlist"], bin_settings[0])

    data.update({
        "sinds" : sinds,
        "fullcat": fullcat,
        "bounds": bounds,
        "plpairs": plpairs,
    })
    if "jkey" in params["lenskey"].keys() and params["lenskey"]["jkey"] is not None:
        data.update({"jk" : fullcat[params["lenskey"]["jkey"]]})
    return data


class XIO(object):
    def __init__(self, lenses, randoms=None, params=None, dirpaths=None, nrandoms=None,
                 force_centers=100):
        """
        XSHEAR style input file creator

        Parameters
        ----------
        lenses : dict
            dictionary with lens data
        randoms : dict, optional
            dictionary with random points data
        params : dict
            Pipeline settings in a dictionary format.
            If :code:`None` then the default :py:data:`paths.params` will be used
        dirpaths : dict
            Pipeline directory paths in a dictionary format.
            If :code:`None` then the default :py:data:`paths.dirpaths` will be used
        nrandoms : float or int
            number of random points to draw for each parameter bin with replacement.
            If :code:`None` then the value is extracted from :py:data:`paths.params`
        force_centers : int or np.array
            number of JackKnife centers, or the (RA, DEC) positions of the centers

        Examples
        --------

        Using default parameters specified in :py:data:`paths.params`.

        load catalogs::

            lenses = parbins.prepare_lenses()
            randoms = parbins.prepare_random()

        initiate object::

            xio = parbins.XIO(lenses, randoms)

        create project directory::

            xio.mkdir()

        loop over all parameter bins::

            xio.loop_bins(norands=args.norands)

        write logfile, for future reference::

            logfile = xio.dpath + '/' + paths.params['tag'] + '_params.p'
            pickle.dump(paths.params, open(logfile, 'wb'))

        """

        # containers
        self.flist = []
        self.rlist = []
        self.flist_jk = []
        self.rlist_jk = []
        self.ind = 0
        self.bin_tag = None

        self.lenses = lenses
        self.randoms = randoms

        # inds of random points drawn
        self.idraw = None

        if nrandoms is None:
            self.nrandoms = paths.params["nrandoms"][paths.params["mode"]]
        else:
            self.nrandoms = nrandoms

        if params is None and dirpaths is None:
            self.params = paths.params
            self.dirpaths = paths.dirpaths
        elif None in (params, dirpaths):
            raise SyntaxError("Some of the arguments are left at default,"
                              " which results in inconsistent behaviour. "
                              "If using custom input, define all arguments!")
        else:
            self.params = params
            self.dirpaths = dirpaths

        self.centers = force_centers
        if "jk" in lenses.keys() and "jk" in lenses.keys():
            self.centers = None

        self._bin_jk_centers = None

        # set random seed
        self.rng = np.random.RandomState()
        self.rng.seed(seed=self.params["seeds"]['random_seed'])

    def setbin(self, bound):
        """Assigns index and filenames for param bin based on :code:`bounds`"""
        self.bin_tag = "_qbin"
        for bb in bound:
            self.bin_tag += "-" + str(bb[2])

        self.flist.append(self.params["tag"] + "_" + self.params["lens_prefix"] +
                          self.bin_tag + ".dat")
        self.rlist.append(self.params["tag"] + "_" + self.params["rand_prefix"] +
                          self.bin_tag + ".dat")

    def mkdir(self):
        """creates project directory"""
        self.dpath = self.dirpaths['xin'] + "/" + self.params["tag"]
        if not os.path.isdir(self.dpath):
            os.mkdir(self.dpath)

    def _save_jk_cens(self):
        fname = self.dpath + '/' + self.params["tag"] + "_jkcens" + self.bin_tag + '.dat'
        np.savetxt(fname, self._bin_jk_centers)

    def save_clust(self):
        """Writes lenses to file in xshear style"""
        sind = self.lenses['sinds'][self.ind]
        makecat(self.dpath + "/" + self.flist[self.ind],
                        self.lenses['id'][sind], self.lenses['ra'][sind],
                        self.lenses['dec'][sind], self.lenses['z'][sind])
        print 'saved ' + self.flist[self.ind]

    def save_clust_jk(self):
        """writes cluster lens file for each JK patch in xshear style"""
        sind = self.lenses["sinds"][self.ind]
        ra = self.lenses["ra"][sind]
        dec = self.lenses["dec"][sind]

        if self.centers is not None:
            cens = self.centers
            # assuming number of JK centers is specified
            if not np.iterable(self.centers):
                cens = np.min((self.centers, len(ra)))
            _labels, self._bin_jk_centers = assign_kmeans_labels(np.vstack((ra, dec)).T, cens)
            jkinds, jkninds, labels = assign_jk_labels(ra, dec, self._bin_jk_centers)
            self._save_jk_cens()
        else:
            print "here"
            jkinds, jkninds, labels = extract_jk_labels(self.lenses["jk"][sind])

        # write data table of selected clusters along with the assigned Jackknife IDs
        ftab = to_pandas(self.lenses["fullcat"][sind])
        ftab["JK_ID"] = labels
        fio.write(self.dpath + "/" + self.flist[self.ind].replace('.dat', '.fits'), ftab.to_records(), clobber=True)

        for label, jkind in enumerate(jkinds):
            fname = self.dpath + "/" +  self.flist[self.ind].replace('.dat', '_patch' + str(label) + '.dat')
            self.flist_jk.append(fname)
            makecat(fname, self.lenses["id"][sind][jkind], self.lenses["ra"][sind][jkind],
                    self.lenses["dec"][sind][jkind], self.lenses["z"][sind][jkind])

    def randsel(self, match=True):
        """
        Selects random points to use (weighted draw with replacement)

        If :code:`self.nrandoms == -1` then all random points are used, and no random draw is made.
        In this case the weights are not applied properly, so be careful!

        Parameters
        ----------
        match : bool
            Flag to match the random point distribution by their paramters to
            the lens distribution. The alternative is to just apply the parameter cut without
            further matching.
        """

        rind = self.randoms['sinds'][self.ind]
        pars = self.randoms["qlist"][rind]
        sind = self.lenses['sinds'][self.ind]
        refpars = self.lenses["qlist"][sind]

        rw = self.randoms['w'][rind]
        if match:
            rw = matchdd(pars, refpars, win=rw)
        prw = rw / rw.sum()

        nr = rind.sum()
        if self.nrandoms == -1:
            self.idraw = np.arange(nr)
        else:
            self.idraw = self.rng.choice(np.arange(nr), size=self.nrandoms, p=prw, replace=True)


    def save_rands(self):
        """Writes random points to file in xshear style"""
        rind = self.randoms['sinds'][self.ind]
        makecat(self.dpath + "/" + self.rlist[self.ind],
                        self.randoms['id'][rind][self.idraw], self.randoms['ra'][rind][self.idraw],
                        self.randoms['dec'][rind][self.idraw], self.randoms['z'][rind][self.idraw])
        print 'saved ' + self.rlist[self.ind]

    def save_rands_jk(self):
        """writes random points to file for each JK patch in xshear style"""
        rind = self.randoms['sinds'][self.ind]
        ra = self.randoms['ra'][rind][self.idraw]
        dec = self.randoms['dec'][rind][self.idraw]

        if self.centers is not None:
            jkinds, jkninds, labels = assign_jk_labels(ra, dec, self._bin_jk_centers)
        else:
            jkinds, jkninds, labels = extract_jk_labels(self.randoms["jk"][rind][self.idraw])

        for label, jkind in enumerate(jkinds):
            fname = self.dpath + "/" + self.rlist[self.ind].replace('.dat', '_patch' + str(label) + '.dat')
            self.rlist_jk.append(fname)
            makecat(fname, self.randoms["id"][rind][jkind], self.randoms["ra"][rind][jkind],
                    self.randoms["dec"][rind][jkind], self.randoms["z"][rind][jkind])

    def loop_bins(self, norands=False, match=True):
        """
        Loops over (\lambda, z) parameter bins and save xshear input files

        Parameters
        ----------
        norands : bool
            Flag to skip random points, default :code:`False`, (that is to include randoms)
        match : bool
            Flag to match the random point distribution by their paramters to
            the lens distribution. The alternative is to just apply the parameter cut without
            further matching.
        """

        # loop over parameter bins
        for ind, bound in enumerate(self.lenses["bounds"]):
            self.ind = ind
            self.setbin(bound)

            self.save_clust()
            self.save_clust_jk()

            if not norands:
                self.randsel(match=match)
                self.save_rands()
                self.save_rands_jk()

            break


def assign_kmeans_labels(pos, centers, verbose=False):
    """
    Defines 2D patches on the sky via spherical k-means

    Parameters
    ----------
    pos : np.ndarray
        positions of points in (RA, DEC)
    centers : int or np.ndarray
        Number of centers to use, or the (RA, DEC) coordinates of the centers
    verbose : bool
        verbose flag to pass to **kmeans_radec**

    Returns
    -------
    np.array, np.ndarray
        K-means labels, K-means centers
    """

    if not np.iterable(centers):  # if centers is a number
        ncen = centers
        nsample = pos.shape[0] // 2
        km = krd.kmeans_sample(pos, ncen=ncen,
                                    nsample=nsample, verbose=verbose)
        if not km.converged:
            km.run(pos, maxiter=100)
    else:  # if centers is an array of RA, DEC pairs
        assert len(centers.shape) == 2  # shape should be (:, 2)
        km = krd.KMeans(centers)

    labels = km.find_nearest(pos).astype(int)
    return labels, km.centers


def assign_jk_labels(ra, dec, centers):
    """
    Assigns a Jacknife (JK) label to the points based on the passed centers

    Parameters
    -----------
    ra : np.array
        Right Ascension of objects
    dec : np.array
        Declination of objects
    centers : np.array
        Coordinates of centers for K-means patches

    Returns
    --------
    bool array, bool array, int array
        * inds which are *NOT IN* patch i,
        * inds which are *IN* patch i,
        * JK labels
    """

    pos = np.vstack((ra, dec)).T

    km = krd.KMeans(centers)

    labels = km.find_nearest(pos).astype(int)


    sub_labels = np.arange(len(centers), dtype=int)
    # sub_labels = np.unique(labels)

    # indexes of clusters for subsample i
    non_indexes = [np.where(labels != ind)[0] for ind in sub_labels]

    # indexes of clusters not in subsample i
    indexes = [np.where(labels == ind)[0] for ind in sub_labels]

    return indexes, non_indexes, labels


def extract_jk_labels(labels):
    """
    Extracts JK-labels from k-means label array

    Parameters
    ----------
    labels : np.array
        k-means labels

    Returns
    -------
    bool array, bool array, int array
        * inds which are *NOT IN* patch i,
        * inds which are *IN* patch i,
        * JK labels

    """

    sub_labels = np.unique(labels.astype(int))

    # indexes of clusters for subsample i
    non_indexes = [np.where(labels != ind)[0] for ind in sub_labels]

    # indexes of clusters not in subsample i
    indexes = [np.where(labels == ind)[0] for ind in sub_labels]
    return indexes, non_indexes, labels