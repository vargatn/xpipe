"""
Extracts pwsum files int PDFContainers
"""


import numpy as np
import argparse


import xpipe.paths as paths
import xpipe.xhandle.pzboost as pzboost
import xpipe.xhandle.xwrap as xwrap
import xpipe.xhandle.parbins as parbins

parser = argparse.ArgumentParser(description='Runs xshear with the rotated sources mode')
parser.add_argument('--noclust', action="store_true", default=False)
parser.add_argument('--norands', action="store_true", default=False)
parser.add_argument('--ibin', type=int, default=None)

if __name__ == "__main__":
    args = parser.parse_args()

    flist, flist_jk, rlist, rlist_jk = parbins.get_file_lists(paths.params, paths.dirpaths)

    if not args.noclust:
        # combining clusters
        pairs_files, bin_vals = xwrap.extract_pairs_bins(flist_jk)
        infodicts = pzboost.create_infodicts(pairs_names=pairs_files, bin_vals=bin_vals)

        if args.ibin is not None:
            pzboost.combine_pwsums(infodicts[args.ibin])
        else:
            for info in infodicts:
                pzboost.combine_pwsums(info)


    if not args.norands:
        # combining randoms
        pairs_files, bin_vals = xwrap.extract_pairs_bins(rlist_jk)
        infodicts = pzboost.create_infodicts(pairs_names=pairs_files, bin_vals=bin_vals)

        if args.ibin is not None:
            pzboost.combine_pwsums(infodicts[args.ibin])
        else:
            for info in infodicts:
                pzboost.combine_pwsums(info)
