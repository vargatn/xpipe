"""
Reads shear catalog and rotates shears randomly using full METACALIBRATION responses

Be sure to run --wconfig first! only runs the measurement if --wconfig=False


Useful command for IBM LOADLEVELER:
llq | grep di49quq | awk '{print $1}' | xargs -L1 llhold -r
"""

from __future__ import print_function
import argparse

import numpy as np
import xpipe.paths as paths
import xpipe.xhandle.xwrap as xwrap
import xpipe.xhandle.parbins as parbins


parser = argparse.ArgumentParser(description='Runs xshear with the rotated sources mode')
parser.add_argument('--head', action="store_true", default=False)
parser.add_argument('--nchunks', type=int)
parser.add_argument('--ichunk', type=int)
parser.add_argument('--params', type=str)
parser.add_argument('--nometa', action="store_false", default=True)
parser.add_argument('--noclust', action="store_true")
parser.add_argument('--norands', action="store_true")



if __name__ == '__main__':
    args = parser.parse_args()
    paths.update_params(args.params)
    print("starting calculation")

    # reading lists for bins
    print("starting rotations...")
    flist, flist_jk, rlist, rlist_jk = parbins.get_file_lists(paths.params, paths.dirpaths)
    alist = []
    if not args.noclust:
        alist += flist
    if not args.norands:
        alist += rlist

    if args.nchunks is not None and args.nchunks > 0 and args.ichunk is not None:
        print('chunkwise rotation')
        print('calculating rotated shear for ' + str(args.ichunk) + '/' + str(args.nchunks) + ' chunk')
        xwrap.chunkwise_rotate(alist, metasel=args.nometa, nrot=paths.params['shear_nrot'],
                               nchunks=args.nchunks, ichunk=args.ichunk,
                               head=args.head, seed_master=paths.params["seeds"]['shear_seed_master'])
    else:
        print('serial rotation')
        xwrap.serial_rotate(alist, metasel=args.nometa, nrot=paths.params['shear_nrot'], head=args.head,
                            seed_master=paths.params["seeds"]['shear_seed_master'])

