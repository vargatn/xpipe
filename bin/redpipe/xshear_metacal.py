"""
Wrapper to run xshear with METACALIBRATION shears

USAGE:
run 5 times in total:
 - first for shear responses and shear,
 - 4 times to calculate inputs for selection responses

"""

import argparse
import numpy as np

import xpipe.paths as paths
import xpipe.xhandle.parbins as parbins
import xpipe.xhandle.xwrap as xwrap

parser = argparse.ArgumentParser(description='Runs xshear')
parser.add_argument('--head', type=int, default=0)
parser.add_argument('--nopairs', action="store_false")
parser.add_argument('--noclust', action="store_true")
parser.add_argument('--norands', action="store_true")
parser.add_argument('--params', type=str)

shape_path = paths.fullpaths[paths.params["shear_to_use"]]
sheared_shape_paths = []
for tag in xwrap.sheared_tags:
    sheared_shape_paths.append(shape_path.replace(".dat", tag + ".dat"))

if __name__ == '__main__':
    args = parser.parse_args()
    paths.update_params(args.params)

    xpath = parbins.get_dpath(paths.params, paths.dirpaths) + "/" +\
            paths.params["tag"] + "_xconfig.cfg"
    xwrap.write_xconf(xpath)

    flist, flist_jk, rlist, rlist_jk = parbins.get_file_lists(paths.params, paths.dirpaths)

