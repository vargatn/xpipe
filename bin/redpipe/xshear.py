"""
Wrapper to run xshear
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

if __name__ == '__main__':
    args = parser.parse_args()
    paths.update_params(args.params)

    xpath = parbins.get_dpath(paths.params, paths.dirpaths) + "/" +\
            paths.params["tag"] + "_xconfig.cfg"
    xwrap.write_xconf(xpath)

    flist, flist_jk, rlist, rlist_jk = parbins.get_file_lists(paths.params, paths.dirpaths)

    print flist_jk
    # TODO this is placeholder
    alist = np.concatenate(flist_jk)
    # blist = np.concatenate(rlist_jk)
    # alist = np.concatenate(flist_jk[3:8] + flist_jk[11:16] + flist_jk[19:])
    # blist = np.concatenate(rlist_jk[3:8] + rlist_jk[11:16] + rlist_jk[19:])

    if not args.noclust:
        clust_infos = xwrap.create_infodict(alist, head=args.head,
                                            pairs=args.nopairs, src_bins=paths.params["source_bins_to_use"],
                                            xconfig=xpath)
        xwrap.multi_xrun(clust_infos, nprocess=paths.params['nprocess'])


    if not args.norands:
        rands_infos = xwrap.create_infodict(blist, head=args.head,
                                            pairs=args.nopairs, src_bins=paths.params["source_bins_to_use"],
                                            xconfig=xpath)
        xwrap.multi_xrun(rands_infos, nprocess=paths.params['nprocess'])
