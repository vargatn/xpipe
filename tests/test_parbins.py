import pytest
import numpy as np
import fitsio as fio
import glob
import xpipe.xhandle.parbins as parbins
import yaml


_params = {'custom_data_path': None,
       'pdf_paths': None,
       'mode': 'dev',
       'tag': 'pytest',
       'shear_style': 'metacal',
       'shear_to_use': 'pytest',
       'cat_to_use': 'pytest',
       'source_bins_to_use': [2, 3],
       'param_bins_full': {'q0_edges': [0.1, 0.5, 1.0], 'q1_edges': [20, 40, 100.0]},
       'param_bins_dev': {'q0_edges': [0.1, 1.0], 'q1_edges': [20, 100]},
       'nprocess': 3,
       'njk_max': 100,
       'nrandoms': {'full': -1, 'dev': 701},
       'seeds': {'random_seed': 5, 'shear_seed_master': 10},
       'headsize': 5000000,
       'cosmo_params': {'H0': 70.0, 'Om0': 0.3},
       'radial_bins': {'nbin': 15, 'rmin': 0.1, 'rmax': 100, 'units': 'Mpc'},
       'weight_style': 'uniform',
       'pairlog': {'pairlog_rmin': 0, 'pairlog_rmax': 15, 'pairlog_nmax': 10000},
       'lenskey': {'id': 'MEM_MATCH_ID',
                   'ra': 'RA',
                   'dec': 'DEC',
                   'z': 'Z_LAMBDA',
                   'q0': 'Z_LAMBDA',
                   'q1': 'LAMBDA_CHISQ'},
       'randkey': {'q0': 'ZTRUE',
                   'q1': 'AVG_LAMBDAOUT',
                   'ra': 'RA',
                   'dec': 'DEC',
                   'z': 'ZTRUE',
                   'w': 'WEIGHT'},
       'lens_prefix': 'y3lens',
       'rand_prefix': 'y3rand',
       'fields_to_use': None,
       'pzpars': {'hist': {'nbin': 15, 'zmin': 0.0, 'zmax': 3.0, 'tag': 'zhist'},
                  'full': {'tag': 'zpdf'},
                  'boost': {'rbmin': 3, 'rbmax': 13}}}


# @pytest.fixture(scope="session")
def mock_lens_catalog():

    np.random.seed(seed = 5)
    nrows = 631 # a prime number
    ids = np.arange(nrows, dtype=int)
    ra = np.random.uniform(low=0., high=40., size=nrows)
    dec = np.random.uniform(low=-30., high=10., size=nrows)
    z = np.random.uniform(low=0.1, high=1.0, size=nrows)

    richness = np.random.uniform(low=10, high=150, size=nrows)

    raw_data = np.zeros(nrows, dtype=[("MEM_MATCH_ID", "i8"), ("RA", "f8"), ("DEC", "f8"), ("LAMBDA_CHISQ", "f8"), ("Z_LAMBDA", "f8")])
    raw_data["MEM_MATCH_ID"] = ids
    raw_data["RA"] = ra
    raw_data["DEC"] = dec
    raw_data["LAMBDA_CHISQ"] = richness
    raw_data["Z_LAMBDA"] = z

    _edges = _params["param_bins_dev"]
    ii = ((_edges["q0_edges"][0] < raw_data["Z_LAMBDA"]) * (_edges["q0_edges"][1] > raw_data["Z_LAMBDA"]) *
          (_edges["q1_edges"][0] < raw_data["LAMBDA_CHISQ"]) * (_edges["q1_edges"][1] > raw_data["LAMBDA_CHISQ"]))

    dev_data = raw_data[ii]

    return raw_data, dev_data

# @pytest.fixture(scope="session")
def mock_rand_catalog():

    np.random.seed(seed = 5)
    nrows = 1701 # a prime number
    ids = np.arange(nrows, dtype=int)
    ra = np.random.uniform(low=0., high=40., size=nrows)
    dec = np.random.uniform(low=-30., high=10., size=nrows)
    z = np.random.uniform(low=0.1, high=1.0, size=nrows)

    richness = np.random.uniform(low=10, high=160, size=nrows)

    raw_data = np.zeros(nrows, dtype=[("ID", "i8"), ("RA", "f8"), ("DEC", "f8"), ("AVG_LAMBDAOUT", "f8"), ("ZTRUE", "f8"), ("WEIGHT", "f8")])
    raw_data["ID"] = ids
    raw_data["RA"] = ra
    raw_data["DEC"] = dec
    raw_data["AVG_LAMBDAOUT"] = richness
    raw_data["ZTRUE"] = z
    raw_data["WEIGHT"] = np.ones(len(z))

    _edges = _params["param_bins_dev"]
    ii = ((_edges["q0_edges"][0] < raw_data["ZTRUE"]) * (_edges["q0_edges"][1] > raw_data["ZTRUE"]) *
          (_edges["q1_edges"][0] < raw_data["AVG_LAMBDAOUT"]) * (_edges["q1_edges"][1] > raw_data["AVG_LAMBDAOUT"]))

    dev_data = raw_data[ii]

    return raw_data, dev_data

@pytest.fixture(scope="session")
def mock_data_folder(tmpdir_factory):

    _path_lenscat = str(tmpdir_factory.mktemp("lenscat"))
    _path_in = str(tmpdir_factory.mktemp("xshear_in"))
    _path_out = str(tmpdir_factory.mktemp("xshear_out"))
    _path_results = str(tmpdir_factory.mktemp("results"))

# tmpdir_factory.mktemp("lenscat")

    raw_data, dev_lens_data = mock_lens_catalog()
    fio.write(_path_lenscat + "/pytest_lens.fits", raw_data, clobber=True)
    raw_data, dev_rand_data = mock_rand_catalog()
    fio.write(_path_lenscat + "/pytest_rand.fits", raw_data, clobber=True)

    fullpaths = {
        "pytest": {
            "lens": _path_lenscat + "/pytest_lens.fits",
            "rand": _path_lenscat + "/pytest_rand.fits",
        }
    }

    dirpaths = {
        "lenscat": _path_lenscat,
        "xin": _path_in,
        "xout": _path_out,
        "results": _path_results
    }

    return fullpaths, dirpaths, dev_lens_data, dev_rand_data



def test_data_loading(mock_data_folder):
    fullpaths, dirpaths, dev_lens_data, dev_rand_data = mock_data_folder
    data, lenscat = parbins.load_lenscat(_params, fullpaths)


def test_rand_loading(mock_data_folder):
    fullpaths, dirpaths, dev_lens_data, dev_rand_data = mock_data_folder
    data, randcat = parbins.load_randcat(_params, fullpaths)


def test_prepare_lenses(mock_data_folder):
    fullpaths, dirpaths, dev_lens_data, dev_rand_data = mock_data_folder
    data = parbins.prepare_lenses(params=_params, fullpaths=fullpaths)

    assert len(data["fullcat"]) == 631
    assert len(data["sinds"]) == 1
    assert np.sum(data["sinds"][0]) == 367

def test_prepare_randoms(mock_data_folder):
    fullpaths, dirpaths, dev_lens_data, dev_rand_data = mock_data_folder
    data = parbins.prepare_random(params=_params, fullpaths=fullpaths)

    assert len(data["fullcat"]) == 1701
    assert len(data["sinds"]) == 1
    assert np.sum(data["sinds"][0]) == 937

# print(len(lenscat), len(dev_lens_data))
    # assert len(lenscat) == len(dev_lens_data)

def test_xio_lens(mock_data_folder):
    fullpaths, dirpaths, dev_lens_data, dev_rand_data = mock_data_folder
    lenses = parbins.prepare_lenses(params=_params, fullpaths=fullpaths)
    # randoms = parbins.prepare_random(params=_params, fullpaths=fullpaths)

    xio = parbins.XIO(lenses, None, force_centers=75, params=_params, dirpaths=dirpaths)
    xio.mkdir()

    xio.loop_bins(norands=True)

    tmp = fio.read(dirpaths["xin"] + "/pytest/pytest_y3lens_qbin-0-0.fits")
    assert len(tmp) == 367
    tmp = glob.glob(dirpaths["xin"] + "/pytest/pytest_y3lens_qbin-0-0*patch*.dat")
    assert len(tmp) == 75


def test_xio_lens_rand(mock_data_folder):
    fullpaths, dirpaths, dev_lens_data, dev_rand_data = mock_data_folder
    lenses = parbins.prepare_lenses(params=_params, fullpaths=fullpaths)
    randoms = parbins.prepare_random(params=_params, fullpaths=fullpaths)

    xio = parbins.XIO(lenses, randoms, force_centers=75, params=_params, dirpaths=dirpaths)
    xio.mkdir()

    xio.loop_bins(norands=False)
    tmp = fio.read(dirpaths["xin"] + "/pytest/pytest_y3rand_qbin-0-0.fits")
    assert len(tmp) == 937
    tmp = glob.glob(dirpaths["xin"] + "/pytest/pytest_y3rand_qbin-0-0*patch*.dat")
    assert len(tmp) == 75


def test_file_list(mock_data_folder):
    fullpaths, dirpaths, dev_lens_data, dev_rand_data = mock_data_folder

    flist, flist_jk, rlist, rlist_jk = parbins.get_file_lists(params=_params, dirpaths=dirpaths)

    true_flist = ["pytest_y3lens_qbin-0-0.dat",]
    true_rlist = ["pytest_y3rand_qbin-0-0.dat",]
    true_flist_jk = [["pytest_y3lens_qbin-0-0_patch{}.dat".format(i) for i in np.arange(75)],]
    true_rlist_jk = [["pytest_y3rand_qbin-0-0_patch{}.dat".format(i) for i in np.arange(75)],]

    assert flist == true_flist
    assert rlist == true_rlist

    assert flist_jk == true_flist_jk
    assert rlist_jk == true_rlist_jk
#