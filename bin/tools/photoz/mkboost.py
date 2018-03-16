"""
Extracts P(z) decomposition boost factors from pdf files
"""

import numpy as np
import pickle
import argparse

import xpipe.paths as paths
import xpipe.xhandle.pzboost as pzboost
import xpipe.xhandle.xwrap as xwrap
import xpipe.xhandle.parbins as parbins

parser = argparse.ArgumentParser(description='extracts boost factors from logged source-lens pairs')
parser.add_argument('--tag', type=str, default="")


def write_pz_boost(prof_dict, fname):
    """Extracts correlation boost factor"""
    profheader = "R\t1+B(z)\tB(z)_err"
    res = np.vstack((prof_dict['boost_rvals'], prof_dict['boost'], prof_dict['boost_err'])).T
    np.savetxt(fname, res, header=profheader)

    fname2 = fname.replace(".dat", "_cov.dat")
    covheader = "Covariance of gaussian amplitudes"
    rescov = prof_dict['boost_cov']
    np.savetxt(fname2, rescov, header=covheader)


def get_bname(pnames):
    root_name = pnames[0].split("_result_pairs")[0]
    if "_patch" in root_name:
        root_name = root_name.split("_patch")[0]
    bname = root_name + pzboost.raw_pdf_tag + paths.params["pzpars"]["full"]['tag'] + '.p'
    return bname


if __name__ == "__main__":
    args = parser.parse_args()
    print "Starting Boost factor calculation in *full* mode"

    flist, flist_jk, rlist, rlist_jk = parbins.get_file_lists(paths.params, paths.dirpaths)

    pairs_files, bin_vals = xwrap.extract_pairs_bins(flist_jk)

    for i, pnames in enumerate(pairs_files):

        bname = get_bname(pnames)
        print "processing", bname

        bcont = pzboost.PDFContainer.from_file(bname)
        bcont.decomp_gauss(refprof=paths.params['pzpars']['boost']['refprof'],
                           force_rbmin=paths.params['pzpars']['boost']['rbmin'],
                           force_rbmax=paths.params['pzpars']['boost']['rbmax'])
        boostdict = bcont.to_boostdict()

        bpath = (bname.split("/")[-1].split("_pzcont")[0] + "_" +
                 paths.params['pzpars']["full"]['tag'] + '_boostdict' + args.tag + '.p')

        print 'saving '
        print bpath
        pickle.dump(boostdict, open(bpath, 'wb'))

        respath = (bname.split("/")[-1].split("_pzcont")[0] + "_" +
                   paths.params['pzpars']["full"]['tag'] + '_boost' + args.tag + '.dat')

        write_pz_boost(boostdict,  respath)
        print respath
        print 'finished'


