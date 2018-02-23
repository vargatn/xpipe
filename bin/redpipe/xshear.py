"""
Wrapper to run xshear
"""

import argparse
import numpy as np

import xpipe.paths as paths
import xpipe.xhandle.parbins as parbins
import xpipe.xhandle.xwrap as xwrap
import xpipe.tools.selector as selector

parser = argparse.ArgumentParser(description='Runs xshear')
parser.add_argument('--head', type=int, default=0)
parser.add_argument('--nopairs', action="store_false")
parser.add_argument('--noclust', action="store_true")
parser.add_argument('--norands', action="store_true")
parser.add_argument('--params', type=str)

parser.add_argument("--nchunk", type=int, default=1)
parser.add_argument("--ichunk", type=int, default=0)


if __name__ == '__main__':
    args = parser.parse_args()
    paths.update_params(args.params)

    xpath = parbins.get_dpath(paths.params, paths.dirpaths) + "/" +\
            paths.params["tag"] + "_xconfig.cfg"
    xwrap.write_xconf(xpath)

    flist, flist_jk, rlist, rlist_jk = parbins.get_file_lists(paths.params, paths.dirpaths)

    _alist = np.concatenate(flist_jk)
    alist = selector.partition(_alist, args.nchunk)[args.ichunk]

    _blist = np.concatenate(rlist_jk)
    blist = selector.partition(_blist, args.nchunk)[args.ichunk]

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
