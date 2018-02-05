"""
Divides the redMaPPer cluster and randoms catalog into bins in richness and redshift (and JK regions)
"""

import argparse
import pickle
import proclens.paths as paths
import proclens.xhandle.parbins as parbins


parser = argparse.ArgumentParser(description='Creates parameter bins for xshear calculations')
parser.add_argument('--params', type=str)
parser.add_argument('--norands', default=False, action="store_true")

if __name__ == '__main__':
    args = parser.parse_args()
    paths.update_params(args.params)

    param_bins, nrandoms = paths.get_bin_settings(paths.params, devmode=paths.devmode)
    logfile = paths.dirpaths['xin'] + '/' + paths.params['tag'] + '_params.p'
    pickle.dump(paths.params, open(logfile, 'wb'))


    # lpars = parbins.prepare_lenses(param_bins)


