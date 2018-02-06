"""
Handles calling xshear with multiprocessing (OpenMP-style single node calculations)
"""

import os
import copy
import numpy as np
import subprocess as sp
import multiprocessing as mp
from collections import OrderedDict
import pandas as pd
import fitsio as fio
import scipy.interpolate as interp

from .. import paths

BADVAL = -9999.
np.seterr('warn')


###################################################################
# xshear config file generation

head = OrderedDict([
    ('H0', 70.),
    ('omega_m', 0.3),
    ('healpix_nside', 64),
    ('mask_style', 'none'),
])


def get_shear(params=None):
    """
    Returns the shear config part of an *XSHEAR* config file

    Parameters
    ----------
    params : dict
        Pipeline settings in a dictionary format. If :code:`None` then the default
        :py:data:`paths.params` will be used

    Returns
    -------
    dict
        shear settings
    """
    if params is None:
        params = paths.params

    shear = OrderedDict([
        ('shear_style', params['shear_style']),
        ('sigmacrit_style', 'sample'),
        ('shear_units', 'both'),
        ('sourceid_style',  'index'),
    ])
    return shear


def get_redges(params=None):
    """
    Returns the radial binning config part of an *XSHEAR* config file

    Parameters
    ----------
    params : dict
        Pipeline settings in a dictionary format. If :code:`None` then the default
        :py:data:`paths.params` will be used

    Returns
    -------
    dict
        radial binning settings
    """

    if params is None:
        params = paths.params

    redges = OrderedDict([
        ('rmin', params['radial_bins']['rmin']),
        ('rmax', params['radial_bins']['rmax']),
        ('nbin', params['radial_bins']['nbin']),
        ('r_units', params['radial_bins']['units'])
    ])
    return redges


def get_pairlog(params=None):
    """
    Returns the source-lens pair logging config part of an *XSHEAR* config file

    Parameters
    ----------
    params : dict
        Pipeline settings in a dictionary format. If :code:`None` then the default
        :py:data:`paths.params` will be used

    Returns
    -------
    dict
        source-lens pair logging settings
    """

    if params is None:
        params = paths.params

    pairlog = OrderedDict([
        ('pairlog_rmin', params['pairlog']['pairlog_rmin']),
        ('pairlog_rmax', params['pairlog']['pairlog_rmax']),
        ('pairlog_nmax', params['pairlog']['pairlog_nmax']),
    ])
    return pairlog


pairlog_nopairs = OrderedDict([
    ('pairlog_rmin', 0),
    ('pairlog_rmax', 0),
    ('pairlog_nmax', 0),
])

tail = OrderedDict([
    ('zdiff_min', 0.1),
])

tail_uniform = OrderedDict([
    ('zdiff_min', 0.1),
    ('weight_style', "uniform"),
])


def get_default_xshear_settings(params=None):
    """The baseline settings for a Metacal lensing measurement"""
    default_xshear_settings = {
        'head': head,
        'shear': OrderedDict([
            ('shear_style', 'metacal'),
            ('sigmacrit_style', 'sample'),
            ('shear_units', 'both'),
            ('sourceid_style', 'index'),
        ]),
        'redges': get_redges(params=params),
        'pairlog': pairlog_nopairs,
        'tail': tail,
    }
    return default_xshear_settings


def addlines(cfg, odict):
    """appends lines to config file"""
    for key in odict.keys():
        line = key + ' = ' + str(odict[key]) + '\n'
        cfg.write(line)


def write_custom_xconf(fname, xsettings=None, params=None):
    """
    Writes custom *XSHEAR* config file

    Parameters
    ----------
    fname : str
        path name for config file
    xsettings : dict
        custom settings
    params : dict
        Pipeline settings in a dictionary format.
        If :code:`None` then the default :py:data:`paths.params` will be used

    """

    settings = get_default_xshear_settings(params=params)
    if xsettings is not None:
        settings.update(xsettings)
    with open(fname, 'w+') as cfg:
        addlines(cfg, settings['head'])
        addlines(cfg, settings['shear'])
        addlines(cfg, settings['redges'])
        addlines(cfg, settings['pairlog'])
        addlines(cfg, settings['tail'])


def write_xconf(fname, pairs=True):
    """
    Writes simple *XSHEAR* config file

    Parameters
    ----------
    fname : str
        path name for config file    pairs
    pairs : bool
        flag to log source-lens pairs

    """

    with open(fname, 'w+') as cfg:
        addlines(cfg, head)
        addlines(cfg, get_shear())
        addlines(cfg, get_redges())
        if pairs:
            addlines(cfg, get_pairlog())
        else:
            addlines(cfg, pairlog_nopairs)
        addlines(cfg, tail)

###################################################################


def get_main_source_settings(params=None):
    """Load settings for unsheared METACAL run *with* pairlogging"""
    if params is None:
        params = paths.params
    main_source_settings = {
        "shear": OrderedDict([
            ('shear_style', "metacal"),
            ('sigmacrit_style', 'sample'),
            ('shear_units', 'both'),
            ('sourceid_style', 'index'),
        ]),
        "pairlog": get_pairlog(params=params)
    }
    return main_source_settings


def get_main_source_settings_nopairs():
    """Load settings for unsheared METACAL run *with out* pairlogging"""
    main_source_settings_nopairs = {
        "shear": OrderedDict([
            ('shear_style', "metacal"),
            ('sigmacrit_style', 'sample'),
            ('shear_units', 'both'),
            ('sourceid_style', 'index'),
        ]),
        "pairlog": pairlog_nopairs
    }
    return main_source_settings_nopairs


sheared_tags = ["_1p", "_1m", "_2p", "_2m"]
sheared_source_settings = {
    "shear": OrderedDict([
        ('shear_style', "reduced"),
        ('sigmacrit_style', 'sample'),
        ('shear_units', 'both'),
        ('sourceid_style', 'index'),
    ]),
    "pairlog": pairlog_nopairs
}


###################################################################
# rotated shear catalog


def rot2d(e1, e2, alpha):
    """2D counterclockwise roation matrix"""
    e1n = e1 * np.cos(alpha) - e2 * np.sin(alpha)
    e2n = e1 * np.sin(alpha) + e2 * np.cos(alpha)
    return e1n, e2n


class CatRotator(object):
    def __init__(self, fname, seed=5, e_inds=(3, 4)):
        """
        Loads shear catalog, and saves a randomly rotated version

        Parameters
        ----------
        fname : str
            path to the shear catalog (ascii format)
        seed : int
            random seed for rotation
        e_inds : list
            inxed for :c;Pode:`e1` and :code:`e2` column of the shear catalog

        Notes
        -----

        Steps:

            * read catalog from ascii file

            * rotate catalog based on random seed

            * write catalog to file. The output file name is defined by the prefix: :code:`rot_seed' + str(self.seed) + '_'`

        The rotated catalog file is later deleted after the calculations finished

        """

        self.fname = fname
        self.seed = seed
        self.rng = np.random.RandomState(seed=self.seed)
        self.cat = None
        self.get_oname()

        if len(e_inds) == 2 and e_inds[1] - e_inds[0] == 1:
            self.ind_e1 = e_inds[0]
            self.ind_e2 = e_inds[1]
        else:
            raise ValueError('e_inds must contain two consecutive integer values indexing e1, and e2')

    def get_oname(self):
        """get output name"""
        head, tail = os.path.split(self.fname)
        self.oname = head + '/rot_seed' + str(self.seed) + '_' + tail

    def _rotcat(self):
        """apply random rotation to shears"""
        e1 = self.cat[:, self.ind_e1]
        e2 = self.cat[:, self.ind_e2]

        alpha = self.rng.uniform(low=0.0, high=2. * np.pi, size=len(e1))
        e1n, e2n = rot2d(e1, e2, alpha)
        self.cat[:, self.ind_e1] = e1n
        self.cat[:, self.ind_e2] = e2n

    def readcat(self):
        """read shear catalog"""
        self.cat = pd.read_csv(self.fname, sep=' ', header=None).values

    def writecat(self):
        """write *rotated* shear catalog"""
        fmt = ['%d', ] + (self.cat.shape[1] - 1) * ['%.6f']
        np.savetxt(self.oname, self.cat, fmt=fmt)

    def rmcat(self):
        """delete shear catalog"""
        try:
            os.remove(self.oname)
        except OSError:
            pass

    def rotate(self):
        """
        Perform a random rotation on the shear catalog
        """
        print '    starting reading'
        self.readcat()
        print '    rotating with seed = ' + str(self.seed)
        self._rotcat()
        print '    writing...'
        self.writecat()
        print '    finished'

