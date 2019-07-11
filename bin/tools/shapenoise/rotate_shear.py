"""
Reads shear catalog and rotates shears randomly using full METACALIBRATION responses

Be sure to run --wconfig first! only runs the measurement if --wconfig=False
"""

import argparse

import xpipe.paths as paths
import xpipe.xhandle.xwrap as xwrap

parser = argparse.ArgumentParser(description='Runs xshear with the rotated sources mode')
parser.add_argument('--wconfig', action="store_true", default=False)
parser.add_argument('--head', action="store_true", default=False)
parser.add_argument('--nchunks', type=int)
parser.add_argument('--ichunk', type=int)
parser.add_argument('--params', type=str)
parser.add_argument('--nometa', type=bool, action="store_false")


if __name__ == '__main__':
    args = parser.parse_args()
    paths.update_params(args.params)
    print "starting calculation"

    ismeta = np.invert(args.nometa)

    if args.wconfig:
        print "writing config files"
        main_xconfig_fname = paths.fullpaths["xconfig"]
        xwrap.write_custom_xconf(main_xconfig_fname, xsettings=xwrap.get_main_source_settings_nopairs())
        if ismeta:
            for i, tag in enumerate(xwrap.sheared_tags):
                sheared_xconfig_fname = main_xconfig_fname.replace("xconfig", "xconfig" + tag)
                xwrap.write_custom_xconf(sheared_xconfig_fname, xsettings=xwrap.sheared_source_settings)

    else:
        # reading lists for bins
        print "starting rotations..."
        with open(paths.fullpaths['flist']) as file:
            flist = file.read().splitlines()

        with open(paths.fullpaths['rlist']) as file:
            rlist = file.read().splitlines()

        alist = flist + rlist

        if args.nchunks is not None and args.nchunks > 0 and args.ichunk is not None:
            print 'chunkwise rotation'
            print 'calculating rotated shear for ' + str(args.ichunk) + '/' + str(args.nchunks) + ' chunk'
            xwrap.chunkwise_rotate(alist, ismeta=ismeta, nrot=paths.params['shear_nrot'],
                                   nchunks=args.nchunks, ichunk=args.ichunk,
                                   head=args.head, seed_master=paths.params['shear_seed_master'])
        else:
            print 'serial rotation'
            xwrap.serial_rotate(alist, ismeta=ismeta, nrot=paths.params['shear_nrot'], head=args.head,
                                seed_master=paths.params['shear_seed_master'])

