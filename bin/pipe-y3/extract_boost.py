
import pandas as pd
import numpy as np

import xpipe.paths as paths
import xpipe.xhandle.parbins as parbins
import xpipe.xhandle.pzboost as pzboost

import astropy.cosmology as cosmology

import pickle

import xpipe.tools.y3_sompz as sompz
import xpipe.likelihood.quintiles as quintiles

import argparse

fname_root = "/e/ocean1/users/vargatn/DES/pairs/"

if __name__ == "__main__":

    main_file_path = "/e/ocean1/users/vargatn/DESY3/Y3_mastercat_03_31_20.h5"
    src = sompz.sompz_reader(main_file_path)
    src.build_lookup()

    flist, flist_jk, rlist, rlist_jk = parbins.get_file_lists(paths.params, paths.dirpaths)

    for i, tmp in enumerate(flist):
        print(i)
        smb = pzboost.SOMBoost(src, [flist_jk[i],])
        fname_pairs = fname_root + flist[i].replace(".dat", "_pairs.p")
        smb.prep_boost(bins_to_use=np.linspace(0, 14, 15), pair_outpath=fname_pairs)

