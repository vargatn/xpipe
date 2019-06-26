"""
Rename and Blind xshear  DeltaSigma profiles as used for Lensing Without Borders
"""

from __future__ import print_function, division
import argparse
import copy
import os
import numpy as np

import xpipe.paths as paths
import xpipe.xhandle.parbins as parbins
import xpipe.xhandle.blinding as blinding

parser = argparse.ArgumentParser(description='postprocesses xshear output')
parser.add_argument('--params', type=str)

#LWB_BLIND_SEED = 8472
LWB_BLIND_SEED = 8473
blinder = blinding.BlindLWB(seed=LWB_BLIND_SEED)

if __name__ == '__main__':
    args = parser.parse_args()
    paths.update_params(args.params)

    flist, flist_jk, rlist, rlist_jk = parbins.get_file_lists(paths.params, paths.dirpaths)
    for i, clust_files in enumerate(flist_jk):
        print(i)
        bin_tag = clust_files[0].split("_" + paths.params["lens_prefix"])[1].split("_patch")[0]

        resroot = paths.dirpaths["results"] + "/" +\
                  paths.params["tag"] + "/" + paths.params["tag"] + "_" + paths.params["subtr_prefix"] + bin_tag

        full_main_name = paths.fullpaths["lwb"]["lens"].flatten()[i]
        tmp, main_name =  os.path.split(full_main_name)
        oname_root = paths.dirpaths["results"] + "/"  + paths.params["tag"] + "/" +\
                     main_name.replace(".fits", "_subtracted_wsys_blind")

        subtr_prof = np.loadtxt(resroot + "_profile.dat")
        subtr_dst_cov = np.loadtxt(resroot + "_dst_cov.dat")
        subtr_dsx_cov = np.loadtxt(resroot + "_dsx_cov.dat")

        blinder.draw()
        factor = blinder.f(subtr_prof[:, 0])

        subtr_prof_blind = subtr_prof.copy()
        subtr_prof[:, 1] *= factor

        # Saving profile
        profheader = "<R> [Mpc]\tDeltaSigma_t [M_sun / pc^2]\tDeltaSigma_t_err [M_sun / pc^2]\tDeltaSigma_x [M_sun / pc^2]\tDeltaSigma_x_err [M_sun / pc^2]"

        print("saving:", oname_root)
        np.savetxt(oname_root + "_profile.dat", subtr_prof, header=profheader)

        # Saving covariance
        np.savetxt(oname_root + "_dst_cov.dat", subtr_dst_cov)
        np.savetxt(oname_root + "_dsx_cov.dat", subtr_dsx_cov)



