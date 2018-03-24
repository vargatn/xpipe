"""
Extracts PDF-s into pwsum files by matching pairs files
"""

import numpy as np
import argparse

import xpipe.paths as paths
import xpipe.xhandle.pzboost as pzboost
import xpipe.xhandle.xwrap as xwrap
import xpipe.xhandle.parbins as parbins


parser = argparse.ArgumentParser(description='Extract distributions from xshear output')
parser.add_argument('--ichunk', type=int, default=None)
parser.add_argument('--nchunk', type=int, default=None)
parser.add_argument('--ibin', type=int, default=None)
parser.add_argument('--noclust', action="store_true", default=False)
parser.add_argument('--norands', action="store_true", default=False)
parser.add_argument('--force_rbin', type=int, default=None)
parser.add_argument('--pdfid', type=str, default="coadd_objects_id")
parser.add_argument('--run_missing', action='store_true', default=False)


force_zcens = np.linspace(0.01, 3.50, 350)

if __name__ == "__main__":
    args = parser.parse_args()

    flist, flist_jk, rlist, rlist_jk = parbins.get_file_lists(paths.params, paths.dirpaths)

    if not args.noclust:
        # exectuting cluster chunk
        pairs_files, bin_vals = xwrap.extract_pairs_bins(flist_jk)
        raw_infodicts = pzboost.create_infodicts(pairs_names=pairs_files,
                                                 bin_vals=bin_vals,
                                                 pdf_paths=paths.pdf_files,
                                                 pdfid=args.pdfid,
                                                 force_zcens=force_zcens,
                                                 force_rbin=args.force_rbin)

        infodicts = pzboost.balance_infodicts(raw_infodicts, args.ibin, args.nchunk, args.ichunk)
        if args.run_missing:
            pnames = [info["pname"] for info in infodicts]
            fnames, haspairs = pzboost.check_pwsum_files(pnames)
            infodicts = infodicts[np.invert(haspairs)]
        pzboost.multi_pwsum_run(infodicts, nprocess=paths.params['nprocess'])

    if not args.norands:
        # exectuting random points chunk
        pairs_files, bin_vals = xwrap.extract_pairs_bins(rlist_jk)
        raw_infodicts = pzboost.create_infodicts(pairs_names=pairs_files,
                                                 bin_vals=bin_vals,
                                                 pdf_paths=paths.pdf_files,
                                                 pdfid=args.pdfid,
                                                 force_zcens=force_zcens,
                                                 force_rbin=args.force_rbin)

        infodicts = pzboost.balance_infodicts(raw_infodicts, args.ibin, args.nchunk, args.ichunk)
        if args.run_missing:
            pnames = [info["pname"] for info in infodicts]
            fnames, haspairs = pzboost.check_pwsum_files(pnames)
            infodicts = infodicts[np.invert(haspairs)]
        pzboost.multi_pwsum_run(infodicts, nprocess=paths.params['nprocess'])



