"""
 screen -dm bash -c "echo 'hello' > myscript.sh"



"""




import pandas as pd
import numpy as np

import xpipe.paths as paths
import xpipe.xhandle.parbins as parbins

import astropy.cosmology as cosmology

import pickle

import xpipe.tools.y3_sompz as sompz
import xpipe.likelihood.quintiles as quintiles

import argparse

cosmo = cosmology.FlatLambdaCDM(Om0=0.3, H0=70)

paths.update_params("//home/moon/vargatn/DES/PROJECTS/xpipe/settings/params_y3rm_quintiles_meta.yml")

parser = argparse.ArgumentParser(description='which chunk')
parser.add_argument("--lbin", default=None)
parser.add_argument("--zbin", default=None)
parser.add_argument("--nofit", action="store_false", default=True)
parser.add_argument("--noboost", action="store_false", default=True)

parser.add_argument("--runall", action="store_true", default=False)
parser.add_argument("--fiducial", action="store_true", default=False)
parser.add_argument("--refs", action="store_true", default=False)
parser.add_argument("--effs", action="store_true", default=False)
parser.add_argument("--exts", action="store_true", default=False)
parser.add_argument("--feats", action="store_true", default=False)
parser.add_argument("--opts", action="store_true", default=False)
parser.add_argument("--no_overwrite", action="store_false", default=True)

RSEL = [0.004334224019486334, 0.008003474068894987, 0.01080724408713881, 0.011162349600167654]
MS = 1 / (1 + np.array([-0.024,-0.037]))
MS_ERR = 0.008

####################################################
main_file_path = "/e/ocean1/users/vargatn/DESY3/Y3_mastercat_03_31_20.h5"

root_path = "/e/ocean1/users/vargatn/QUINTILES/"

TAG = "lean-fit_effs_v11-lowR"
OPT_FILE_TAG = "opt-score-v1"
features_to_calculate = ["MAGSUM", "BCG_MAGABS_R", "LGAP_SOFT_2", "RGAP_SOFT_2"]

SCALES = (0.1, 3)
SBINS = ((2, 3), (2, 3), (3,))
MS_VALS = (MS, MS, np.array((MS[1],)))

####################################################

if __name__ == "__main__":
    args = parser.parse_args()
    do_fit = args.nofit
    _include_boost = args.noboost
    print("running chunk", str(args.lbin))
    print("fit:",do_fit, "boost:", _include_boost)
    if not args.no_overwrite:
        print("NO OVERWRITE")
        print(args.no_overwrite)

    src = sompz.sompz_reader(main_file_path)
    src.build_lookup()

    features = pd.read_hdf(root_path + "allz_rm_gt5_features.h5", key="data")
    flist, flist_jk, rlist, rlist_jk = parbins.get_file_lists(paths.params, paths.dirpaths)

    i = 0
    for z, zbin in enumerate((0, 1, 2)):
        for l, lbin in enumerate((0, 1, 2, 3, 4, 5, 6)):
            cond = ((args.lbin is None) or (int(args.lbin) == lbin)) & ((args.zbin is None) or (int(args.zbin) == zbin))
            if cond:
                fname_pairs = "/e/ocean1/users/vargatn/DES/pairs/" + flist[i].replace(".dat", "_pairs.p")
                file_tag = root_path + "runs/autosplit_" + TAG + "_z" + str(zbin) + "-l" + str(lbin)
                print(flist[i])

                ms_stds = quintiles.get_mfactor_stds(MS_VALS[zbin], MS_ERR, flist_jk[i])
                # print(ms_stds)
                QE = quintiles.QuintileExplorer(src, flist[i], flist_jk[i], sbins=SBINS[zbin],
                                                pairs_to_load=fname_pairs, file_tag=file_tag, nstep=800,
                                                scales=SCALES, Rs_sbins=RSEL, ms_sbins=MS_VALS[zbin], ms_stds=ms_stds)
                QE.load_target()
                QE.set_features(features)

                if args.runall or args.fiducial:
                    QE.calc_fiducial_profile(do_fit=do_fit, _include_boost=_include_boost)
                    pos = QE.flat_samples.mean(axis=0)
                else:
                    point_means = pickle.load(open(file_tag + "_default_profile.p", "rb"))
                    pos = point_means["flat_samples"].mean(axis=0)
                print(pos)
                QE.init_pos = pos
                QE.nstep = 400
                QE.discard = 100

                if args.runall or args.refs:
                    QE.calc_ref_profiles(do_fit=do_fit, _include_boost=_include_boost, overwrite=args.no_overwrite)

                if args.runall or args.exts:
                    for f, feat_name in enumerate(features_to_calculate):
                        feat1 = QE.feats[:, 0]
                        feat2 = QE.feats[:, np.where(feat_name == features.columns)[0] - 1][:, 0]
                        feat2 = feat2 * np.sign(np.corrcoef(feat1, feat2)[1, 0])
                        QE.calc_custom_expand_profiles(feat1, feat2,
                                                       do_fit=do_fit, _include_boost=_include_boost,
                                                       tag=str(feat_name) + "-expand", overwrite=args.no_overwrite)

                if args.runall or args.feats:
                    QE.calc_feat_profiles(do_fit=do_fit, _include_boost=_include_boost, overwrite=args.no_overwrite)

                if args.runall or args.effs:
                    QE.calc_eff_profiles(do_fit=do_fit, _include_boost=_include_boost, overwrite=args.no_overwrite)

                if args.runall or args.opts:
                    fname = file_tag + "_" + OPT_FILE_TAG + ".p"
                    feat = pickle.load(open(fname, "rb"))
                    QE.calc_custom_profiles(feat, do_fit=do_fit, _include_boost=_include_boost, tag=OPT_FILE_TAG,
                                            overwrite=args.no_overwrite)

            i += 1

