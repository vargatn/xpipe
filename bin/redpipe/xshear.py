"""
Wrapper to run xshear
"""

import argparse


import proclens.paths as paths
import proclens.xhandle.xwrap as xwrap

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
    xpath = paths.get_dpath(paths.params, paths.dirpaths) + "/" +\
            paths.params["tag"] + "_xconfig.cfg"
    xwrap.write_xconf(xpath)


    if not args.noclust:
        pass

    if not args.norands:
        pass