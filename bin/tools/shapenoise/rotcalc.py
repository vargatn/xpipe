"""
Step 3
"""
import argparse
import numpy as np

import xpipe.paths as paths
import xpipe.xhandle.xwrap as xwrap


parser = argparse.ArgumentParser(description='Process rotated shape results')
parser.add_argument('--nmax', type=int)
parser.add_argument('--dev_tag', type=str, default="")
parser.add_argument('--params', type=str)

if __name__ == '__main__':
    args = parser.parse_args()
    paths.update_params(args.params)
    print args.dev_tag

    seed_rots = xwrap.get_rot_seeds(nrot=paths.params['shear_nrot'],
                                    seed_master=paths.params['shear_seed_master'])
    seeds_to_use = seed_rots
    if args.nmax is not None:
        seeds_to_use = seed_rots[:args.nmax]
    print len(seeds_to_use)
    ledges, zedges, nrandoms = paths.get_bin_settings(paths.params, devmode=paths.devmode)

    mean_dst = []
    mean_covs = []
    clust_names, rand_names, bin_vals = xwrap.get_result_files(ledges, zedges, npatch=None)
    for i, (clust_name, rand_name) in enumerate(zip(clust_names, rand_names)):

        rootpath = paths.dirpaths['results'] + '/' + paths.subtr_prefix + \
                   '_l' + str(bin_vals[i][0]) + '_z' + str(bin_vals[i][1])

        dsts = []
        covs = []
        for s, seed in enumerate(seeds_to_use):
        # read xshear output files
        #     print seed, i

            seedpath = rootpath + '_seed' + str(seed)
            dstpath = seedpath + '_profile.dat'
            covpath = seedpath + '_dst_cov.dat'

            dsts.append(np.loadtxt(dstpath))
            covs.append(np.loadtxt(covpath))

        mean_covs.append(np.median(covs, axis=0))
        mean_dst.append(np.median(dsts, axis=0))

        cname = rootpath + args.dev_tag + '_shapecov.dat'
        print cname
        np.savetxt(cname, mean_covs[-1])

        cname = rootpath + args.dev_tag + '_shapedst.dat'
        np.savetxt(cname, mean_dst[-1])

