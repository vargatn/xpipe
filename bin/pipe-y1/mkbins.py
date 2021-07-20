"""
Divides the lens and random points catalog into bins by parameters (and JK regions)
"""

import argparse
import pickle
import xpipe.paths as paths
import xpipe.xhandle.parbins as parbins


parser = argparse.ArgumentParser(description='Creates parameter bins for xshear calculations')
parser.add_argument('--params', type=str)
parser.add_argument('--norands', default=False, action="store_true")


if __name__ == '__main__':
    args = parser.parse_args()
    paths.update_params(args.params)

    lenses = parbins.prepare_lenses()
    if not args.norands:
        randoms = parbins.prepare_random()
    else:
        randoms = None

    xio = parbins.XIO(lenses, randoms, force_centers=paths.params["njk_max"])
    xio.mkdir()

    logfile = xio.dpath + '/' + paths.params['tag'] + '_params.p'
    pickle.dump(paths.params, open(logfile, 'wb'))

    xio.loop_bins(norands=args.norands)

