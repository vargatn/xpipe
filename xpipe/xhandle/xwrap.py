"""
Handles calling xshear with multiprocessing (OpenMP-style single node calculations)
"""

import os
import numpy as np
import subprocess as sp
import multiprocessing as mp
from collections import OrderedDict
import pandas as pd

from ..tools.selector import partition
from .. import paths

BADVAL = -9999.
np.seterr('warn')


###################################################################
# xshear config file generation

def get_head(params=None):
    """
    Returns the cosmology config part of an *XSHEAR* config file

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

    head = OrderedDict([
        ('H0', params["cosmo_params"]["H0"]),
        ('omega_m', params["cosmo_params"]["Om0"]),
        ('healpix_nside', 64),
        ('mask_style', 'none'),
    ])
    return head


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
        ('weight_style', params['weight_style']),
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
        'head': get_head(params=params),
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
    Writes simple *XSHEAR* config file based on :py:data:`paths.params`

    Parameters
    ----------
    fname : str
        path name for config file    pairs
    pairs : bool
        flag to log source-lens pairs

    """

    with open(fname, 'w+') as cfg:
        addlines(cfg, get_head())
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


def get_metanames(fnames):
    metanames = []
    for tag in sheared_tags:
        _fnames = []
        for fname in fnames:
            _fnames.append(fname.replace(".dat", tag + ".dat"))
        metanames.append(_fnames)
    return metanames


###################################################################
# rotated shear catalog


def get_rot_seeds(nrot, seed_master):
    """Radnom generates seeds for random rotations using the master seed"""
    rng = np.random.RandomState(seed=seed_master)
    smax = np.iinfo(np.uint32).max
    seed_rots = rng.randint(low=0, high=smax, size=nrot)
    return seed_rots


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
        """Perform a random rotation on the shear catalog"""
        print '    starting reading'
        self.readcat()
        print '    rotating with seed = ' + str(self.seed)
        self._rotcat()
        print '    writing...'
        self.writecat()
        print '    finished'


def single_rotate(flist, seed_rot, metasel=False, head=False):
    """
    runs one single rotation of the source catalog with METACAL SELECTION RESPONSES

    Parameters
    ----------
    flist : list
        list of individual *xshear* input files
    seed_rot : int
        random seed to be used in the rotation
    metasel : bool
        whether to perform the rotation in calculating the metacal selection responses also.
    head : bool
        whether to use only the first few rows of the shear catalog

    """
    #"""runs one single rotation of the source catalog with METACAL SELECTION RESPONSES"""

    print "    writing xshear input"
    xpath = paths.dirpaths["xin"] + "/" + paths.params["tag"] + \
            "xconfig_rot_seed" + str(seed_rot) + ".cfg"
    write_custom_xconf(xpath, xsettings=get_main_source_settings_nopairs())

    print '    creating main catalog'
    sname = paths.fullpaths[paths.params['shear_to_use']]
    cr = CatRotator(sname, seed=seed_rot)
    cr.rotate()
    print '    running measurement'
    clust_infos = create_infodict(flist, pairs=False, head=head,
                                  rotate=True, seed=seed_rot, shape_path=cr.oname,
                                  xconfig=xpath)

    multi_xrun(clust_infos, nprocess=paths.params['nprocess'])
    print '    finished...'
    print '    removing catalog'
    cr.rmcat()

    # starting runs for metacal selection responses
    if metasel:
        print '    creating sheared catalogs'
        for tag in sheared_tags:
            xpath_sheared = paths.dirpaths["xin"] + "/" + paths.params["tag"] + \
                            "xconfig" + tag + "_rot_seed" + str(seed_rot) + ".cfg"
            write_custom_xconf(xpath_sheared, xsettings=sheared_source_settings)

            sname = paths.fullpaths[paths.params['shear_to_use']].replace(".dat", tag + ".dat")
            cr = CatRotator(sname, seed=seed_rot)
            cr.rotate()
            print '    running sheared measurement', tag
            clust_infos = create_infodict(flist, pairs=False, head=head,
                                          rotate=True, seed=seed_rot, shape_path=cr.oname, metatag=tag,
                                          xconfig=xpath_sheared)
            multi_xrun(clust_infos, nprocess=paths.params['nprocess'])
            print '    finished...'
            print '    removing catalog'
            cr.rmcat()


def serial_rotate(flist, metasel=False, nrot=20, head=False, seed_master=5):
    """
    performs the random rotations serially and saves them to file


    Parameters
    ----------
    flist : list
        list of individual *xshear* input files
    metasel : bool
        whether to perform the rotation in calculating the metacal selection responses also.
    nrot : int
        number of rotations to perform
    head : bool
        whether to use only the first few rows of the shear catalog
    seed_master : int
        master seed to generate random seeds for the rotating the entire catalog

    """

    seed_rots = get_rot_seeds(nrot=nrot, seed_master=seed_master)

    for i, seed_rot in enumerate(seed_rots):
        print 'rotation ', i
        single_rotate(flist, seed_rot, metasel=metasel, head=head)


def chunkwise_rotate(flist, metasel=False, nrot=20, nchunks=1, ichunk=0, head=False, seed_master=5):
    """
    Performs the random rotations and saves them to file

    Parameters
    ----------
    flist : list
        list of individual *xshear* input files
    metasel : bool
        whether to perform the rotation in calculating the metacal selection responses also.
    nrot : int
        number of rotations in total
    nchunks : int
        number of chunks to divide the task into
    ichunk : int
        index of the current chunk
    head : bool
        whether to use only the first few rows of the shear catalog
    seed_master : int
        master seed to generate random seeds for the rotating the entire catalog


    """
    seed_rots = get_rot_seeds(nrot=nrot, seed_master=seed_master)
    seed_chunks = partition(seed_rots, nchunks)

    print 'using seeds: ', seed_chunks[ichunk]
    for i, seed_rot in enumerate(seed_chunks[ichunk]):
        print 'rotation ', i
        single_rotate(flist, seed_rot, metasel=metasel, head=head)


###################################################################


def create_infodict(flist, head=False, pairs=False, seed=None,
                    rotate=False, seed_tag="", shape_path=None, metatag=None,
                    xconfig=None, params=None, dirpaths=None, fullpaths=None, src_bins=(0,)):
    """
    Creates configuration dictionary which can be passed to multiprocessing map_async()

    Parameters
    ----------
    flist : list
        list of individual *xshear* input files
    head : bool
        whether to use only the first few rows of the shear catalog
    pairs : bool
        whether to log source-lens pairs
    seed : int
        random seed
    rotate : bool
        whether to perform random rotations
    seed_tag : str
        additional comment to write to each file. default is ``"_seed" + str(seed)``
    shape_path : str
        absolute path to the source catalog. If None the default path will be loaded
        from :py:data:`path.params`
    metatag : str
        tag to indicate which metacalibration sheared version this run corresponds to
    xconfig : str
        absolute path to xshear executable
    params : dict
        Pipeline settings in a dictionary format.
        If :code:`None` then the default :py:data:`paths.params` will be used
    dirpaths : dict
        Pipeline directory paths in a dictionary format.
        If :code:`None` then the default :py:data:`paths.fullpaths` will be used
    fullpaths : dict
        Pipeline file paths in a dictionary format.
        If :code:`None` then the default :py:data:`paths.fullpaths` will be used

    Returns
    -------
    list of dict

    """

    if params is None and dirpaths is None and fullpaths is None:
        params = paths.params
        dirpaths = paths.dirpaths
        fullpaths = paths.fullpaths
    elif None in (params, dirpaths, fullpaths):
        raise SyntaxError("Some of the arguments are left at default,"
                          " which results in inconsistent behaviour. "
                          "If using custom input, define all arguments!")

    iroot = dirpaths["xin"] + "/" + params["tag"] + "/"
    oroot = dirpaths["xout"] + "/" + params["tag"] + "/"

    if rotate:
        seed_tag += '_seed' + str(seed)

    if metatag is None:
        metatag = ""

    if shape_path is None:
        shape_path = fullpaths[params['shear_to_use']]

    if xconfig is None:
        xconfig = dirpaths["xin"] + "/" + params["tag"] + "/" + params["tag"] + "_xconfig.dat"


    infodicts = []
    for file in flist:
        for isrc in src_bins:

            src_tag = ""
            _spath = shape_path

            if len(src_bins) > 1:
                src_tag = "_sbin" + str(isrc)
                _spath = shape_path + src_tag + ".dat"

            infodict_raw = {
                'infile':  iroot + file,
                'outfile': oroot + file.split('.dat')[0] + src_tag + seed_tag + '_result' + metatag + '.dat',
                'pairsfile': oroot + file.split('.dat')[0] + src_tag + seed_tag + '_result_pairs' + metatag + '.dat',
                'logfile': oroot + file.split('.dat')[0] + src_tag + seed_tag + '_result_log' + metatag + '.dat',
                'head': head,
                'pairs': pairs,
                'seed': seed,
                'rotate_shears': rotate,
                'shape_path': _spath,
                'xconfig': xconfig,
            }

            infodicts.append(infodict_raw)
    return infodicts


def call_xshear(infodict):
    """
    Calls xshear in a single process

    Run is based on the passed information dict

    Parameters
    ----------
    infodict : dict
        A single list element returned from :py:func:`create_infodict`
    """

    infile = infodict['infile']
    outfile = infodict['outfile']
    pairsfile = infodict['pairsfile']
    logfile = infodict['logfile']

    shape_path = infodict['shape_path']

    if 'head' in infodict.keys() and infodict['head']:
        cmd1 = 'head -n ' + str(infodict['head']) + " " + shape_path
    else:
        cmd1 = 'cat ' + shape_path

    if 'xconfig' in infodict.keys() and infodict['xconfig'] is not None:
        xconfig = infodict['xconfig']
    else:
        xconfig = paths.fullpaths['xconfig']
    print xconfig

    cmd2 = paths.fullpaths['xpath'] + ' ' + xconfig + ' ' +\
           infile + ' ' + pairsfile

    print 'processing ' + os.path.split(infile)[1] + ' with ' + os.path.split(shape_path)[1]

    try:
        rfile = open(outfile, 'w+')
        lfile = open(logfile, 'w+')

        p1 = sp.Popen(cmd1.split(' '), stdout=sp.PIPE)
        p2 = sp.Popen(cmd2.split(' '), stdin=p1.stdout, stdout=rfile, stderr=lfile)

        p1.stdout.close()
        p2.communicate()

        rfile.close()
        lfile.close()
        print 'finished: ' + os.path.split(outfile)[1]
    except KeyboardInterrupt:
        raise KeyboardInterrupt


def call_chunks(chunk):
    """Executes serial calculation for each chunk (simple for loop)"""
    try:
        for infodict in chunk:
            call_xshear(infodict)
    except KeyboardInterrupt:
        pass


def multi_xrun(infodicts, nprocess=1):
    """
    OpenMP style parallelization for xshear tasks

    Separates tasks into chunks, and passes each chunk for an independent process
    for serial evaulation via :py:func:`call_chunks`

    Parameters
    ----------
    infodict : dict
        A single list element returned from :py:func:`create_infodict`
    nprocess : int
        Number of processes (cores) to use. Maximum number is always set by ``len(infodicts)``

    """
    # at most as many processes can be used as there are independent tasks...
    if nprocess > len(infodicts):
        nprocess = len(infodicts)

    print 'starting xshear calculations in ' + str(nprocess) + ' processes'
    fparchunks = partition(infodicts, nprocess)
    pool = mp.Pool(processes=nprocess)
    try:
        pp = pool.map_async(call_chunks, fparchunks)
        pp.get(86400)  # apparently this counters a bug in the exception passing in python.subprocess...
    except KeyboardInterrupt:
        print "Caught KeyboardInterrupt, terminating workers"
        pool.terminate()
        pool.join()
    else:
        pool.close()
        pool.join()

