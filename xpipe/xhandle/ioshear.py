"""
I/O related to shear measurements
"""

import os
import numpy as np
import pandas as pd

from .xwrap import sheared_tags


###################################################################
# Lens catalog operations


def makecat(fname, mid, ra, dec, z):
    """
    Write an xshear style lens catalog to file

    Parameters
    ----------
    fname : str
        full path to file
    mid : np.array -- int
        ID of object
    ra : np.array
        RA
    dec : np.array
        DEC
    z : np.array
        redshift
    """

    table = np.vstack((mid, ra, dec, z, np.ones(shape=z.shape))).T
    fmt = ['%d',] + 3 * ['%.6f',] + ['%d',]
    np.savetxt(fname, table, fmt = fmt)


def read_lens_pos(fnames):
    """
    Reads postions based on the list of filenames passed

    Parameters
    ----------
    fname : str
        full paths to output files

    Returns
    -------
    np.array
        ra, dec of lenses
    """

    pos = []
    for fname in fnames:
        tmp = np.loadtxt(fname)
        pos.append(tmp[:, 1:3])

    pos = np.vstack(pos)

    return pos


###################################################################
# BASE READER

def xread(xdata, **kwargs):
    """
    Reader for xshear output if style is set as `both`

    Each row corresponds to the lensing profile of a single lens, average profiles require additional stacking

    The output data is divided into two tables:

        :code:`info` contains values relevant for the lens in general. It has a shape :code:`(N_lens, 3)`.
        the columns are::

            index:      index from lens catalog
            weight_tot: sum of all weights for all source pairs in all radial bins
            totpairs:   total pairs used

        :code:`data` contains values for individual radial bins. It has a shape :code:`(12, N_lens, N_rbin)`,
        the columns are::

            npair_i:        number of pairs in radial bin i
            rsum_i:         sum of weights * sigma_crit_inv * r
            wsum_i:         sum of weights * sigma_crit_inv
            ssum_i:         sum of weights
            dsum_i:         sum of weights * g_t
            osum_i:         sum of weights * g_x
            dsensum_w_i:    sum of weights * sigma_crit_inv * gsens_t
            osensum_w_i:    sum of weights * sigma_crit_inv * gsens_x
            dsensum_s_i:    sum of weights * gsens_t
            osensum_s_i:    sum of weights * gsens_x
            mean_e1:        sum of weights * g_1
            mean_e2:        sum of weights * g_2

        In each case, _i means radial bin i

    Parameters
    ----------
    xdata : np.array
        raw xshear output loaded to np array

    Returns
    -------
    np.ndarray, np.ndarray, dict
        :code:`info`, :code:`data`, column description dictionary
    """
    valnames = {
        "info": ("index", "weight_tot", "totpairs"),
        "data": ("npair_i", "rsum_i", "wsum_i", "ssum_i", "dsum_i", "osum_i", "dsensum_w_i", "osensum_w_i",
                 "dsensum_s_i", "osensum_s_i", "mean_e1", "mean_e2"),
    }

    # calculates number of radial bins used
    bins = (xdata.shape[1] - 3) // 12

    # position indexes
    sid = 3
    pos_npair = 0
    pos_rsum = 1
    pos_wsum = 2
    pos_ssum = 3
    pos_dsum = 4
    pos_osum = 5
    pos_dsensum_w = 6
    pos_osensum_w = 7
    pos_dsensum_s = 8
    pos_osensum_s = 9
    pos_me1 = 10 # mean e1 shear component
    pos_me2 = 11 # mean e2 shear component

    gid = xdata[:, 0]
    weight_tot = xdata[:, 1]
    tot_pairs = xdata[:, 2]

    npair = xdata[:, sid + pos_npair * bins: sid + (pos_npair + 1) * bins]
    rsum = xdata[:, sid + pos_rsum * bins: sid + (pos_rsum + 1) * bins]
    wsum = xdata[:, sid + pos_wsum * bins: sid + (pos_wsum + 1) * bins]
    ssum = xdata[:, sid + pos_ssum * bins: sid + (pos_ssum + 1) * bins]
    dsum = xdata[:, sid + pos_dsum * bins: sid + (pos_dsum + 1) * bins]
    osum = xdata[:, sid + pos_osum * bins: sid + (pos_osum + 1) * bins]
    dsensum_w = xdata[:, sid + pos_dsensum_w * bins: sid + (pos_dsensum_w + 1) * bins]
    osensum_w = xdata[:, sid + pos_osensum_w * bins: sid + (pos_osensum_w + 1) * bins]
    dsensum_s = xdata[:, sid + pos_dsensum_s * bins: sid + (pos_dsensum_s + 1) * bins]
    osensum_s = xdata[:, sid + pos_osensum_s * bins: sid + (pos_osensum_s + 1) * bins]
    me1 = xdata[:, sid + pos_me1 * bins: sid + (pos_me1 + 1) * bins]
    me2 = xdata[:, sid + pos_me2 * bins: sid + (pos_me2 + 1) * bins]

    info = np.vstack((gid, weight_tot, tot_pairs)).T
    data = np.dstack((npair, rsum, wsum, ssum, dsum, osum,
                      dsensum_w, osensum_w, dsensum_s, osensum_s, me1, me2))
    data = np.transpose(data, axes=(2, 0, 1))

    # checking if loading made sense
    assert (info[:, 2] == np.sum(data[0, :, :], axis=1)).all()

    return info, data, valnames


def xpatches(raw_chunks):
    """
    Processes many smaller xshear output files via **xread**

    Outputs a concatenated array of the many smaller files, along with a
    :code:`chunk_ids` assigned to each row to indicate which chunk it originally came from

    Parameters
    ----------
    raw_chunks : list
        list of raw xshear outputs loaded to np array

    Returns
    -------
    np.ndarray, np.ndarray, np.array
        :code:`info`, :code:`data`, :code:`chunk_ids`
    """

    infos = []
    datas = []
    labels = []
    for i, chunk in enumerate(raw_chunks):
        if len(chunk) > 0:
            if len(chunk.shape) == 1:
                info, data, tmp = xread(chunk[np.newaxis, :])
            else:
                info, data, tmp = xread(chunk)

            infos.append(info)
            datas.append(data)
            labels.append(np.ones(info.shape[0]) * i)

    infos = np.vstack(infos)
    datas = np.concatenate(datas, axis=1)
    labels = np.concatenate(labels)

    return infos, datas, labels


###################################################################
# RAW DATA LOADERS


def read_raw(fname):
    """
    Reads xshear output from file

    Parameters
    ----------
    fname : str
        full path to output file

    Returns
    -------
    np.array
        xshear raw output, empty array if file does not exist
    """

    if os.path.isfile(fname) and os.path.getsize(fname) > 0:
        res = pd.read_csv(fname, delim_whitespace=True, header=None).values
    else:
        res = np.array([])

    return res


def read_multiple_raw(fnames):
    """
    Reads xshear output from multiple files, and concatenates them

    Parameters
    ----------
    fnames : list of str
        full paths to output files
    sort : bool
        if True sorts stacked file by first column of output

    Returns
    -------
    list of np.array
        list of xshear raw outputs, empty array if file does not exist
    """

    res = []
    for fname in fnames:
        if os.path.isfile(fname) and os.path.getsize(fname) > 0:
            res.append(pd.read_csv(fname, delim_whitespace=True, header=None).values)
        else:
            res.append(np.array([]))

    return res


def read_sheared_raw(fname):
    """reads raw xshear output from metacal sheared runs"""
    raw_sheared_data = []
    for tag in sheared_tags:
        rname = fname.replace(".dat", tag + ".dat")
        raw_data = read_raw(rname)
        raw_sheared_data.append(raw_data)
    return raw_sheared_data


def read_multiple_sheared_raw(fnames):
    """reads raw xshear output from metacal sheared runs"""
    raw_sheared_data = []
    for tag in sheared_tags:
        rnames = [fname.replace(".dat", tag + ".dat") for fname in fnames]
        raw_data = read_multiple_raw(rnames)
        raw_sheared_data.append(raw_data)
    return raw_sheared_data


###################################################################
# DATA LOADERS

def read_single_bin(fname, metaname=None):
    """
    Reads and interprets xshear output from a single file

    :code:`info` and :code:`data` are passed on from **xread**,
    while :code:`sheared_data` is just a list of :code:`data` tables corresponding to
    metacalibration sheared results.

    Parameters
    ----------
    fname : str
        full path to output file
    metaname : str
        full base path to sheared output file

    Returns
    -------
    np.ndarray, np.ndarray, list of np.ndarray
        :code:`info`, :code:`data`, :code:`sheared_data`

    """
    raw_main = read_raw(fname)

    (info, data, tmp) = xread(raw_main)
    sheared_data = None

    if metaname:
        raw_sheared = read_sheared_raw(metaname)
        sheared_data = [xread(raw_data)[1]
                        for raw_data in raw_sheared]

    return info, data, sheared_data


def read_multiple_bin(fnames, metanames=None):
    """
    Reads and interprets xshear output from many smaller files

    :code:`info` and :code:`data` are passed on from **xread**,
    while :code:`sheared_data` is just a list of :code:`data` tables corresponding to
    metacalibration sheared results.

    :code:`chunk_ids` assigned to each row to indicate which chunk it originally came from

    Parameters
    ----------
    fname : str
        full path to output file
    metaname : str
        full base path to sheared output file

    Returns
    -------
    np.ndarray, np.ndarray, list of np.ndarray, np.array
        :code:`info`, :code:`data`, :code:`sheared_data`, :code:`chunk_ids`

    """
    raw_main = read_multiple_raw(fnames)

    (cinfo, cdata, clabels) = xpatches(raw_main)

    sheared_data = None
    if metanames:
        raw_sheared = read_multiple_sheared_raw(metanames)
        sheared_data = [xpatches(raw_data)[1]
                        for raw_data in raw_sheared]

    return cinfo, cdata, sheared_data, clabels

