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