"""
Wrapper to run xshear
"""

import argparse


import xpipe.paths as paths
import xpipe.xhandle.parbins as parbins
import xpipe.xhandle.xwrap as xwrap

parser = argparse.ArgumentParser(description='Runs xshear')
parser.add_argument('--head', action="store_true")
parser.add_argument('--nopairs', action="store_false")
parser.add_argument('--noclust', action="store_true")
parser.add_argument('--norands', action="store_true")
parser.add_argument('--params', type=str)

if __name__ == '__main__':
    args = parser.parse_args()
    paths.update_params(args.params)

    # write xshear config file
    xpath = parbins.get_dpath(paths.params, paths.dirpaths) + "/" +\
            paths.params["tag"] + "_xconfig.cfg"
    xwrap.write_xconf(xpath)

    flist, flist_jk, rlist, rlist_jk = parbins.get_file_lists(paths.params, paths.dirpaths)

    if not args.noclust:
        clust_infos = xwrap.create_infodict(flist_jk, head=args.head, pairs=args.nopairs)


    if not args.norands:
        pass