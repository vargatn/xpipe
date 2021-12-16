"""
"""

from __future__ import print_function

import copy
import numpy as np
import argparse

import xpipe.xhandle.shearops as shearops
import xpipe.tools.selector as sl

import xpipe.paths as paths
import xpipe.xhandle.xwrap as xwrap
import xpipe.xhandle.parbins as parbins


parser = argparse.ArgumentParser(description='Process rotated shape results')
parser.add_argument('--ichunk', type=int)
parser.add_argument('--nchunk', type=int)
parser.add_argument('--params', type=str)
parser.add_argument('--lensweight', action="store_true", default=False)


def read_pos(flist, root_path):
    poss = []
    for fname in flist:
        name = root_path + fname
        poss.append(np.loadtxt(name)[:, 1:3])
    return poss

dev_tag = ''
ismeta=True
no_metaseed=True


if __name__ == '__main__':
    args = parser.parse_args()
    paths.update_params(args.params)

    seed_rots = xwrap.get_rot_seeds(nrot=paths.params['shear_nrot'],
                                    seed_master=paths.params["seeds"]['shear_seed_master'])

    seeds_to_use = seed_rots
    if args.nchunk is not None and args.ichunk is not None:
        seeds_to_use = sl.partition(seed_rots, args.nchunk)
        seeds_to_use = seeds_to_use[args.ichunk]

    flist, flist_jk, rlist, rlist_jk = parbins.get_file_lists(paths.params, paths.dirpaths)

    xinroot = paths.dirpaths["xin"] + "/" + paths.params["tag"] + "/"
    cpos = read_pos(flist, xinroot)
    rpos = read_pos(rlist, xinroot)

    for s, seed in enumerate(seeds_to_use):
        clust_infos = xwrap.create_infodict(flist, seed=seed, rotate=True)
        rands_infos = xwrap.create_infodict(rlist, seed=seed, rotate=True)
        for i, (cinfo, rinfo) in enumerate(zip(clust_infos, rands_infos)):


            # postprocessing clusters
            clust_file = cinfo["outfile"]

            bin_tag = "_qbin" + clust_file.split("/")[-1].split("qbin")[1].split("_")[0]
            jkname = paths.dirpaths["xin"] + "/" + paths.params["tag"] + "/" + paths.params["tag"] +\
                     "_jkcens" + bin_tag + ".dat"

            centers = np.loadtxt(jkname)
            ncens = len(centers)
            clust_labels = parbins.assign_jk_labels(cpos[i][:, 0], cpos[i][:, 1], centers)[-1]

            weights = None
            if args.lensweight:
                weights = shearops.load_weights(i)

            clust = shearops.process_profile(clust_file, ismeta=ismeta, labels=clust_labels, weights=weights)
            resroot = paths.dirpaths["results"] + "/" + \
                      paths.params["tag"] + "/" + paths.params["tag"] + "_" + paths.params["lens_prefix"] + bin_tag +\
                      "_seed" + str(seed)
            print(resroot)
            xwrap.write_profile(clust, resroot)


            # postprocessing randoms
            rands_file = rinfo["outfile"]

            tag = "_qbin" + clust_file.split("/")[-1].split("qbin")[1].split("_")[0]
            rands_labels = parbins.assign_jk_labels(rpos[i][:, 0], rpos[i][:, 1], centers)[-1]
            rand = shearops.process_profile(rands_file, ismeta=ismeta, labels=clust_labels, weights=weights)
            resroot = paths.dirpaths["results"] + "/" + \
                      paths.params["tag"] + "/" + paths.params["tag"] + "_" + paths.params["rand_prefix"] + bin_tag + \
                      "_seed" + str(seed)
            xwrap.write_profile(rand, resroot)

            prof3 = copy.deepcopy(clust)
            prof3.composite(rand, operation="-")
            resroot = paths.dirpaths["results"] + "/" + \
                      paths.params["tag"] + "/" + paths.params["tag"] + "_" + paths.params["subtr_prefix"] + bin_tag +\
                      "_seed" + str(seed)
            print(resroot)
            xwrap.write_profile(prof3, resroot)
