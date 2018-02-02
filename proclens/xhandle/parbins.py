"""
# TODO add documentation
"""


import numpy as np
from astropy.io import fits
import fitsio as fio

import itertools

from .. import paths

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


def get_fields_auto():
    """Extracts field boundaries from project params.yml"""

    fields = {}
    for name in paths.params["fields_to_use"]:
        fields.update({name : paths.params["fields"][name]})

    return fields


def load_lenscat(fields):
    """
    Loads lens catalog from fits file

    Parameters
    ----------
    fields : dict
        dict of boundaries for :meth:`proclens.xhandle.parbins.field_cut`

    Returns
    -------
    dict, np.array, record-array
        lens catalog data in format,
        bool array for selection,
        raw data table before selection

    Notes
    ------

    The format of the output data table::

        data = {
            "id" : id,
            "ra" : ra,
            "dec" : dec,
            "qlist" : qlist, # np.array of quantities with shape (n_lens, n_quantity)
        }

    """
    lenscat = fits.open(paths.fullpaths[paths.params["cat_to_use"]]["lens"])[1].data
    lenskey = paths.params['lenskey']

    ids = lenscat[lenskey['id']]
    ra = lenscat[lenskey['ra']]
    dec = lenscat[lenskey['dec']]

    select = np.zeros(len(ra), dtype=bool)
    for name in fields.keys():
        select += field_cut(ra, dec, fields[name])

    # number of parameter columns
    nq = len(lenskey.keys()) - 3
    qlist = np.zeros(shape=(len(ra), nq))
    for ival in np.arange(nq):
        colname = "q" + str(ival)
        qlist[:, ival] = lenscat[lenskey[colname]]

    data = {
        "id" : ids,
        "ra" : ra,
        "dec" : dec,
        "qlist" : qlist
    }

    return data, select, lenscat


def load_randcat(fields):
    """
    Loads random point catalog from fits file

    Parameters
    ----------
    fields : list of dicts
        list of boundaries for :meth:`proclens.xhandle.parbins.field_cut`

    Returns
    -------
    dict, np.array, record-array
        lens catalog data in format,
        bool array for selection,
        raw data table before selection

    Notes
    ------

    The format of the output data table::

        data = {
            "id" : id,
            "ra" : ra,
            "dec" : dec,
            "qlist" : qlist, # np.array of quantities with shape (n_lens, n_quantity)
            "w" : w, # weight of random points
        }

    """
    randcat = fits.open(paths.fullpaths[paths.params["cat_to_use"]]["lens"])[1].data
    randkey = paths.params['randkey']

    w = randcat[randkey['w']]
    ra = randcat[randkey['ra']]
    dec = randcat[randkey['dec']]
    ids = len(randcat)

    select = np.zeros(len(ra), dtype=bool)
    for field in fields:
        select += field_cut(ra, dec, field)

    # number of parameter columns
    nq = len(randkey.keys()) - 3
    qlist = np.zeros(shape=(len(ra), nq))
    for ival in np.arange(nq):
        colname = "q" + str(ival)
        qlist[:, ival] = randcat[randkey[colname]]

    data = {
        "id" : ids,
        "w" : w,
        "ra" : ra,
        "dec" : dec,
        "qlist" : qlist
    }

    return data, select, randcat

def prepare_lenses(param_bins):
    pass


def perpare_random(param_bins):
    pass


def log_jk_centers():
    pass


class XIO(object):
    pass

