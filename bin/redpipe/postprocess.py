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
parser.add_argument('--ismeta', action="store_true", default=False)
parser.add_argument('--npatch', default="auto")
parser.add_argument('--params', type=str)
parser.add_argument('--calibs', action="store_true", default=False)
parser.add_argument('--norands', action="store_true", default=False)

# TODO update with intuitive source bin detection
# TODO add better output handling
# TODO add random point subbtraction

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
    flist, flist_jk, rlist, rlist_jk = parbins.get_file_lists(paths.params, paths.dirpaths)
    for i, clust_name in enumerate(flist_jk):
        print "processing bin", i

        clust_infos = xwrap.create_infodict(clust_name)
        clust_files = [info["outfile"] for info in clust_infos]

        clust = shearops.process_profile(clust_files, metanames=None)


        resroot = paths.dirpaths["results"] + "/" +\
                  paths.params["tag"] + "/" + paths.params["lens_prefix"] + "_bin" + str(i)
        write_profile(clust, resroot)
        #
        clusts.append(clust)

        # if not args.norands:
        #     rands_infos = xwrap.create_infodict(rlist_jk[i])
        #     rands_files = [info["outfile"] for info in rands_infos]
        #
        #     rand = shearops.process_profile(rands_files, metanames=None)
        #
        #     resroot = paths.dirpaths["xout"] + "/" + \
        #               paths.params["tag"] + "/" + paths.params["rand_prefix"] + "_bin" + str(i)
        #     write_profile(rand, resroot)
        #
        #     rands.append(rand)
        #
        #     # calculating subtracted profile
        #     resroot = paths.dirpaths['results'] + '/' + paths.subtr_prefix + \
        #               '_l' + str(bin_vals[i][0]) + '_z' + str(bin_vals[i][1])
        #
        #     prof3 = copy.deepcopy(clust)
        #     prof3.composite(rand, operation="-")
        #     xwrap.write_profile(prof3, resroot)



