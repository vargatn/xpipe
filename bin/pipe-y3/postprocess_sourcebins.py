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
if __name__ == '__main__':
    args = parser.parse_args()
    paths.update_params(args.params)

    clusts = []
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

        resroot = paths.dirpaths["results"] + "/" + \
                  paths.params["tag"] + "/" + paths.params["tag"] + "_" + paths.params["lens_prefix"] + bin_tag
        xwrap.write_profile(clust, resroot)

        if args.calibs:
            ccont = xwrap.append_scrit_inv(ccont, clust)