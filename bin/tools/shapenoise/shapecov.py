"""
"""

import copy
import numpy as np
import argparse

import xpipe.xhandle.shearops as shearops
import xpipe.tools.selector as sl

import xpipe.paths as paths
import xpipe.xhandle.xwrap as xwrap


parser = argparse.ArgumentParser(description='Process rotated shape results')
parser.add_argument('--ichunk', type=int)
parser.add_argument('--nchunk', type=int)
parser.add_argument('--params', type=str)


dev_tag = ''
ismeta=True
no_metaseed=True


if __name__ == '__main__':
    args = parser.parse_args()
    paths.update_params(args.params)

    seed_rots = xwrap.get_rot_seeds(nrot=paths.params['shear_nrot'],
                                    seed_master=paths.params['shear_seed_master'])
    seeds_to_use = seed_rots
    if args.nchunk is not None and args.ichunk is not None:
        seeds_to_use = sl.partition(seed_rots, args.nchunk)
        seeds_to_use = seeds_to_use[args.ichunk]

    ledges, zedges, nrandoms = paths.get_bin_settings(paths.params, devmode=paths.devmode)

    with open(paths.fullpaths['flist']) as file:
        flist = file.read().splitlines()
    cpos = xwrap.read_clust_pos(flist)

    with open(paths.fullpaths['rlist']) as file:
        rlist = file.read().splitlines()
    rpos = xwrap.read_clust_pos(rlist)

    clust_names, rand_names, bin_vals = xwrap.get_result_files(ledges, zedges, npatch=None)
    clust_names_patch, rand_names_patch, bin_vals = xwrap.get_result_files(ledges, zedges, npatch="auto")
    for s, seed in enumerate(seeds_to_use):
        # read xshear output files
        for i, (clust_name, rand_name) in enumerate(zip(clust_names, rand_names)):
            print "processing bin", i, "with seed", seed, "which is step", s, "out of", len(seeds_to_use) - 1
            tag = "_seed" + str(seed)

            center_name = paths.dirpaths['xin'] + '/' + paths.params["tag"] + "jkcens" +\
                '_l' + str(bin_vals[i][0]) + '_z' + str(bin_vals[i][1]) + '.dat'
            centers = np.loadtxt(center_name)
            ncens = len(centers)

            # processing clusters
            clust_labels = shearops.get_labels(cpos[i], centers)[0]
            rot_clust_name = clust_name[0].split('_result.dat')[0] + '_seed' + str(seed) + '_result.dat'
            clust = xwrap.process_profile(rot_clust_name, bin_vals[i],
                                          prefix=paths.clust_prefix, labels=clust_labels,
                                          shapemix=True, fnames_fallback=clust_names_patch[i],
                                          ismeta=ismeta, ncens=ncens, no_metaseed=no_metaseed)


            resroot = paths.dirpaths['results'] + '/' + paths.clust_prefix + dev_tag + \
                      '_l' + str(bin_vals[i][0]) + '_z' + str(bin_vals[i][1]) + '_seed' + str(seed)
            print resroot
            xwrap.write_profile(clust, resroot)

            # processing clusters
            rand_labels = shearops.get_labels(rpos[i], centers)[0]
            rot_rand_name = rand_name[0].split('_result.dat')[0] + '_seed' + str(seed) + '_result.dat'
            rand = xwrap.process_profile(rot_rand_name, bin_vals[i],
                                         prefix=paths.rand_prefix, labels=rand_labels,
                                         shapemix=True, fnames_fallback=rand_names_patch[i],
                                         ismeta=ismeta, ncens=ncens, no_metaseed=no_metaseed)


            resroot = paths.dirpaths['results'] + '/' + paths.rand_prefix + dev_tag + \
                      '_l' + str(bin_vals[i][0]) + '_z' + str(bin_vals[i][1]) + '_seed' + str(seed)
            print resroot
            xwrap.write_profile(rand, resroot)

            prof3 = copy.deepcopy(clust)
            prof3.composite(rand, operation="-")

            resroot = paths.dirpaths['results'] + '/' + paths.subtr_prefix + dev_tag + \
                      '_l' + str(bin_vals[i][0]) + '_z' + str(bin_vals[i][1]) + '_seed' + str(seed)
            print resroot
            xwrap.write_profile(prof3, resroot)
