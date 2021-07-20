"""
Collects xshear output files into DeltaSigma profiles
"""
from __future__ import print_function, division
import argparse
import copy
import numpy as np
import pandas as pd
import fitsio as fio

import xpipe.paths as paths
import xpipe.xhandle.parbins as parbins
import xpipe.xhandle.xwrap as xwrap
import xpipe.xhandle.shearops as shearops



parser = argparse.ArgumentParser(description='postprocesses xshear output')
parser.add_argument('--nometa', action="store_false", default=True)
parser.add_argument('--npatch', default="auto")
parser.add_argument('--params', type=str)
parser.add_argument('--calibs', action="store_true", default=False)
parser.add_argument('--norands', action="store_true", default=False)
parser.add_argument('--lensweight', action="store_true", default=False)

# TODO update with intuitive source bin detection, and check with a proper example


def export_scrit_inv(prof):
    """Exports the Sigma_crit inverse from a profile container"""
    refprof = copy.deepcopy(prof)
    refprof.ismeta = False
    refprof.dst_nom = 6
    refprof.dsx_nom = 7
    refprof.dst_denom = 8
    refprof.dsx_denom = 9
    refprof.prof_maker()

    scrit_inv = refprof.dst[-1]
    scrit_inv_err = refprof.dst_err[-1]
    return scrit_inv, scrit_inv_err


if __name__ == '__main__':
    args = parser.parse_args()
    paths.update_params(args.params)

    if args.calibs:
        ccont = xwrap.get_calib_cont()

    clusts = []
    rands = []
    subtrs = []
    flist, flist_jk, rlist, rlist_jk = parbins.get_file_lists(paths.params, paths.dirpaths)
    for i, clust_name in enumerate(flist_jk):
        print("processing bin", i)

        clust_infos = xwrap.create_infodict(clust_name)
        clust_files = [info["outfile"] for info in clust_infos]

        weights = None
        if args.lensweight:
            weights = shearops.load_weights(i)

        bin_tag = clust_files[0].split("_" + paths.params["lens_prefix"])[1].split("_patch")[0]

        clust = shearops.process_profile(clust_files, ismeta=args.nometa, weights=weights)

        resroot = paths.dirpaths["results"] + "/" +\
                  paths.params["tag"] + "/" + paths.params["tag"] + "_" + paths.params["lens_prefix"] + bin_tag
        xwrap.write_profile(clust, resroot)

        if args.calibs:
            ccont = xwrap.append_scrit_inv(ccont, clust)


        if not args.norands:
            rands_infos = xwrap.create_infodict(rlist_jk[i])
            rands_files = [info["outfile"] for info in rands_infos]

            rand = shearops.process_profile(rands_files, ismeta=args.nometa)

            resroot = paths.dirpaths["results"] + "/" + \
                      paths.params["tag"] + "/" + paths.params["tag"] + "_" + paths.params["rand_prefix"] + bin_tag
            xwrap.write_profile(rand, resroot)

            # calculating subtracted profile
            resroot = paths.dirpaths["results"] + "/" + \
                      paths.params["tag"] + "/" + paths.params["tag"] + "_" + paths.params["subtr_prefix"] + bin_tag

            prof3 = copy.deepcopy(clust)
            prof3.composite(rand, operation="-")
            xwrap.write_profile(prof3, resroot)

            rand.drop_data()
            prof3.drop_data()

            rands.append(rand)
            subtrs.append(prof3)

        clust.drop_data()
        clusts.append(clust)

    if args.calibs:
        resroot = paths.dirpaths['results'] + '/' + paths.params["lens_prefix"] + "_calibs.log"# paths.params["calibs_log"]
        bin_vals = []
        print(ccont)
#        for i in np.arange(3):
#           for j in np.arange(7):
#               bin_vals.append((j, i))
#        import pickle
#        pickle.dump(ccont, open("ccont.p", "wb"))
#        xwrap.write_calib_cont(resroot, ccont, bin_vals)


    print("calculating cross-covariance:")
    ccov_t, ccov_x = shearops.stacked_pcov(clusts)
    resroot = paths.dirpaths["results"] + "/" + \
              paths.params["tag"] + "/" + paths.params["tag"] + "_" + paths.params["lens_prefix"] + "_crosscov_dst.dat"
    np.savetxt(resroot, ccov_t)
    resroot = paths.dirpaths["results"] + "/" + \
              paths.params["tag"] + "/" + paths.params["tag"] + "_" + paths.params["lens_prefix"] + "_crosscov_dsx.dat"
    np.savetxt(resroot, ccov_x)

    if not args.norands:
        rcov_t, rcov_x = shearops.stacked_pcov(rands)
        resroot = paths.dirpaths["results"] + "/" +\
                  paths.params["tag"] + "/" + paths.params["tag"] + "_" + paths.params["rand_prefix"] + "_crosscov_dst.dat"
        np.savetxt(resroot, rcov_t)
        resroot = paths.dirpaths["results"] + "/" +\
                  paths.params["tag"] + "/" + paths.params["tag"] + "_" + paths.params["rand_prefix"] + "_crosscov_dsx.dat"
        np.savetxt(resroot, rcov_x)
    #
        scov_t, scov_x = shearops.stacked_pcov(subtrs)
        resroot = paths.dirpaths["results"] + "/" +\
                  paths.params["tag"] + "/" + paths.params["tag"] + "_" + paths.params["subtr_prefix"] + "_crosscov_dst.dat"
        np.savetxt(resroot, scov_t)
        resroot = paths.dirpaths["results"] + "/" +\
                  paths.params["tag"] + "/" + paths.params["tag"] + "_" + paths.params["subtr_prefix"] + "_crosscov_dsx.dat"
        np.savetxt(resroot, scov_x)
