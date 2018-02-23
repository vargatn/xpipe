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
import xpipe.tools.selector as selector

parser = argparse.ArgumentParser(description='Runs xshear')
parser.add_argument('--ichunk', type=int, default=0)
parser.add_argument('--runall', action="store_true")
parser.add_argument('--head', action="store_true")
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

    flist, flist_jk, rlist, rlist_jk = parbins.get_file_lists(paths.params, paths.dirpaths)

    _alist = np.concatenate(flist_jk)
    alist = selector.partition(_alist, args.nchunk)[args.ichunk]

    _blist = np.concatenate(rlist_jk)
    blist = selector.partition(_blist, args.nchunk)[args.ichunk]


    if args.ichunk == 0 or args.runall:
        print xpath
        xwrap.write_custom_xconf(xpath, xsettings=xwrap.get_main_source_settings())


        if not args.noclust:
            clust_infos = xwrap.create_infodict(alist,
                                                head=args.head,
                                                pairs=args.nopairs,
                                                src_bins=paths.params["source_bins_to_use"],
                                                xconfig=xpath)
            
            xwrap.multi_xrun(clust_infos, nprocess=paths.params['nprocess'])

        if not args.norands:
            rands_infos = xwrap.create_infodict(blist,
                                                head=args.head,
                                                pairs=args.nopairs,
                                                src_bins=paths.params["source_bins_to_use"],
                                                xconfig=xpath)
            xwrap.multi_xrun(rands_infos, nprocess=paths.params['nprocess'])

    # this is the sheared computation with flags_select_*
    for i, tag in enumerate(xwrap.sheared_tags):
        if i == (args.ichunk - 1) or args.runall:
            sheared_xconfig_fname = xpath.replace("xconfig", "xconfig" + tag)
            print sheared_xconfig_fname
            xwrap.write_custom_xconf(sheared_xconfig_fname, xsettings=xwrap.sheared_source_settings)

            if not args.noclust:
                sheared_clust_infos = xwrap.create_infodict(alist,
                                                            head=args.head,
                                                            pairs=args.nopairs,
                                                            src_bins=paths.params["source_bins_to_use"],
                                                            xconfig=sheared_xconfig_fname,
                                                            metatag=tag,
                                                            shape_path=sheared_shape_paths[i], )
                xwrap.multi_xrun(sheared_clust_infos, nprocess=paths.params['nprocess'])

            if not args.norands:
                sheared_rand_infos = xwrap.create_infodict(blist,
                                                           head=args.head,
                                                           pairs=args.nopairs,
                                                           src_bins=paths.params["source_bins_to_use"],
                                                           xconfig=sheared_xconfig_fname,
                                                           metatag=tag,
                                                           shape_path=sheared_shape_paths[i], )
                xwrap.multi_xrun(sheared_rand_infos, nprocess=paths.params['nprocess'])