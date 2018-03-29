"""
Collects xshear output files into DeltaSigma profiles
"""

import argparse
import copy
import numpy as np

import xpipe.paths as paths
import xpipe.xhandle.parbins as parbins
import xpipe.xhandle.xwrap as xwrap
import xpipe.xhandle.shearops as shearops



parser = argparse.ArgumentParser(description='postprocesses xshear output')
parser.add_argument('--nometa', action="store_true", default=False)
parser.add_argument('--npatch', default="auto")
parser.add_argument('--params', type=str)
parser.add_argument('--calibs', action="store_true", default=False)
parser.add_argument('--norands', action="store_true", default=False)

# TODO update with intuitive source bin detection, and check with a proper example


def write_profile(prof, path):
    """saves DeltaSigma and covariance in text format"""

    # Saving profile
    profheader = "R [Mpc]\tDeltaSigma_t [M_sun / pc^2]\tDeltaSigma_t_err [M_sun / pc^2]\tDeltaSigma_x [M_sun / pc^2]\tDeltaSigma_x_err [M_sun / pc^2]"
    res = np.vstack((prof.rr, prof.dst, prof.dst_err, prof.dsx, prof.dsx_err)).T
    fname = path + "_profile.dat"
    print "saving:", fname
    np.savetxt(fname, res, header=profheader)

    # Saving covariance
    np.savetxt(path + "_dst_cov.dat", prof.dst_cov)
    np.savetxt(path + "_dsx_cov.dat", prof.dsx_cov)


if __name__ == '__main__':
    args = parser.parse_args()
    paths.update_params(args.params)

    clusts = []
    rands = []
    subtrs = []
    flist, flist_jk, rlist, rlist_jk = parbins.get_file_lists(paths.params, paths.dirpaths)
    for i, clust_name in enumerate(flist_jk):
        print "processing bin", i

        clust_infos = xwrap.create_infodict(clust_name)
        clust_files = [info["outfile"] for info in clust_infos]

        bin_tag = clust_files[0].split("_" + paths.params["lens_prefix"])[1].split("_patch")[0]

        metanames = None
        if not args.nometa:
            metanames = xwrap.get_metanames(clust_files)
        clust = shearops.process_profile(clust_files, metanames=metanames)

        resroot = paths.dirpaths["results"] + "/" +\
                  paths.params["tag"] + "/" + paths.params["tag"] + "_" + paths.params["lens_prefix"] + bin_tag
        write_profile(clust, resroot)

        if not args.norands:
            rands_infos = xwrap.create_infodict(rlist_jk[i])
            rands_files = [info["outfile"] for info in clust_infos]

            metanames = None
            if not args.nometa:
                metanames = xwrap.get_metanames(rands_files)
            rand = shearops.process_profile(rands_files, metanames=metanames)

            resroot = paths.dirpaths["results"] + "/" + \
                      paths.params["tag"] + "/" + paths.params["tag"] + "_" + paths.params["rand_prefix"] + bin_tag
            write_profile(rand, resroot)

            # calculating subtracted profile
            resroot = paths.dirpaths["results"] + "/" + \
                      paths.params["tag"] + "/" + paths.params["tag"] + "_" + paths.params["subtr_prefix"] + bin_tag

            prof3 = copy.deepcopy(clust)
            prof3.composite(rand, operation="-")
            write_profile(prof3, resroot)

            rand.drop_data()
            prof3.drop_data()

            rands.append(rand)
            subtrs.append(prof3)
        
        clust.drop_data()
        clusts.append(clust)


    print "calculating cross-covariance:"
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
