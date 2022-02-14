import pandas as pd
import numpy as np
import healpy as hp
import h5py
import matplotlib.pyplot as plt
import yaml
import fitsio as fio

import copy

from collections import OrderedDict

import xpipe.tools.catalogs as catalogs
import xpipe.paths as paths
import xpipe.xhandle.parbins as parbins
import xpipe.xhandle.xwrap as xwrap
import xpipe.tools.selector as selector
import xpipe.xhandle.shearops as shearops
import matplotlib as mpl
# from matplotlib import rc
mpl.rc('font',**{'family':'serif','serif':['serif']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
mpl.rc('text', usetex=True)

import astropy.constants as constants
import astropy.units as u
import astropy.cosmology as cosmology

import scipy.interpolate as interp

cosmo = cosmology.FlatLambdaCDM(Om0=0.3, H0=70)

from importlib import reload
import pickle

import xpipe.tools.y3_sompz as sompz

import xpipe.tools.selector as sl
import xpipe.tools.visual as visual
import xpipe.xhandle.pzboost as pzboost

import sklearn
import sklearn.covariance
import sklearn.neighbors
import sklearn.decomposition
import scipy.stats as stats

import NFW
import scipy.optimize as optimize
import emcee
from cluster_toolkit import deltasigma

from cluster_toolkit import xi
from cluster_toolkit import bias


from classy import Class
from multiprocessing import Pool

import corner
import xpipe.likelihood.mass as mass
import xpipe.likelihood.quintiles as quintiles

import scipy.ndimage as ndimage
import xpipe.tools.visual as visual
import argparse

TAG = "fit_v4"
do_fit = True
_include_boost = True

parser = argparse.ArgumentParser(description='which chunk')
parser.add_argument("--zbin", type=int, default=0)


if __name__ == "__main__":
    args = parser.parse_args()
    print("running chunk", str(args.zbin))

    main_file_path = "/e/ocean1/users/vargatn/DESY3/Y3_mastercat_03_31_20.h5"
    src = sompz.sompz_reader(main_file_path)
    src.build_lookup()

    features = pd.read_hdf("allz_rm_features.h5", key="data")
    flist, flist_jk, rlist, rlist_jk = parbins.get_file_lists(paths.params, paths.dirpaths)

    #    i = 0
    #    for z, zbin in enumerate((0, 1, 2)):
    #        for l, lbin in enumerate((0, 1, 2, 3)):

    #            if zbin == args.zbin:
    #                fname_pairs = "/e/ocean1/users/vargatn/DES/pairs/" + flist[i].replace(".dat", "_pairs.p")

    #                print(i)
    #                file_tag = "runs/autosplit_" + TAG + "_z" + str(zbin) + "-l" + str(lbin)
    #                print(file_tag)
    #                QE = quintiles.QuintileExplorer(src, flist[i], flist_jk[i], pairs_to_load=fname_pairs, file_tag=file_tag)
    #                QE.load_target()
    #                QE.set_features(features)

    #                QE.calc_fiducial_profile(do_fit=do_fit, _include_boost=_include_boost)
    #        QE.calc_feat_profiles(do_fit=do_fit, _include_boost=_include_boost)
    #        QE.calc_ref_profiles(do_fit=do_fit, _include_boost=_include_boost)

    #            i += 1

    i = 0
    for z, zbin in enumerate((0, 1, 2)):
        for l, lbin in enumerate((0, 1, 2, 3)):
            if zbin == args.zbin:
                fname_pairs = "/e/ocean1/users/vargatn/DES/pairs/" + flist[i].replace(".dat", "_pairs.p")

                print(i)
                file_tag = "runs/autosplit_" + TAG + "_z" + str(zbin) + "-l" + str(lbin)
                print(file_tag)
                QE = quintiles.QuintileExplorer(src, flist[i], flist_jk[i], pairs_to_load=fname_pairs, file_tag=file_tag)
                QE.load_target()
                QE.set_features(features)

                QE.calc_fiducial_profile(do_fit=do_fit, _include_boost=_include_boost)
                QE.calc_feat_profiles(do_fit=do_fit, _include_boost=_include_boost)
            #        QE.calc_ref_profiles(do_fit=do_fit, _include_boost=_include_boost)

            i += 1


    i = 0
    for z, zbin in enumerate((0, 1, 2)):
        for l, lbin in enumerate((0, 1, 2, 3)):
            if zbin == args.zbin:
                fname_pairs = "/e/ocean1/users/vargatn/DES/pairs/" + flist[i].replace(".dat", "_pairs.p")

                print(i)
                file_tag = "runs/autosplit_" + TAG + "_z" + str(zbin) + "-l" + str(lbin)
                print(file_tag)
                QE = quintiles.QuintileExplorer(src, flist[i], flist_jk[i], pairs_to_load=fname_pairs, file_tag=file_tag)
                QE.load_target()
                QE.set_features(features)

                QE.calc_fiducial_profile(do_fit=do_fit, _include_boost=_include_boost)
                #        QE.calc_feat_profiles(do_fit=do_fit, _include_boost=_include_boost)
                QE.calc_eff_profiles(do_fit=do_fit, _include_boost=_include_boost)

            i += 1
