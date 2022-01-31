import pickle
import numpy as np
import pandas as pd
import sklearn

from xpipe.likelihood.mass import make_params, default_cosmo, get_scales
from xpipe.likelihood.mcmc import do_mcmc
from xpipe.xhandle import shearops, pzboost

# TODO update this for Eigen features
# TODO update this for standard features and then correct for their correlations

class QuintileExplorer(object):
    def __init__(self, src, flist, flist_jk, file_tag="autosplit_v1", pairs_to_load=None,
                 z_key="Z_LAMBDA", l_key="LAMBDA_CHISQ", id_key="MEM_MATCH_ID",
                 ismeta=False, bins_to_use=np.linspace(0, 14, 15), npdf=15, **kwargs):
        self.src = src
        self.pair_path = pairs_to_load
        self.z_key = z_key
        self.l_key = l_key
        self.flist = flist
        self.flist_jk = flist_jk
        self.file_tag = file_tag
        self.id_key = id_key
        self.ismeta = ismeta

        self.bins_to_use = bins_to_use
        self.npdf = npdf

        self.lprior = None

        self._quintiles = ((0, 20), (20, 40), (40, 60), (60, 80), (80, 100))

    def load_target(self):
        self.raw_ACP = shearops.AutoCalibrateProfile(self.flist, self.flist_jk, self.src, xlims=(0.1, 100))
        self.raw_ACP.get_profiles(ismeta=self.ismeta)
        self.target = self.raw_ACP.target
        self.smb = pzboost.SOMBoost(self.src, [self.flist_jk,], pairs_to_load=self.pair_path)

    def _calc_profile(self, weights=None, **kwargs):
        ACP = self.raw_ACP.copy()
        ACP.get_profiles(reload=False, ismeta=self.ismeta, weights=weights)

        self.smb.prep_boost(bins_to_use=np.linspace(0, 14, 15))
        self.smb.get_boost_jk(npdf=15, **kwargs)

        ACP.add_boost_jk(self.smb)
        return ACP
    #
    # def _fit_model(self, data, nwalkers=16, params=None, lprior=None, **kwargs) :
    #
    #     if lprior is None:
    #         _lprior = self.lprior
    #
    #     if params is None:
    #         params = self.params
    #     flat_samples = do_mcmc(data, params, nwalkers=nwalkers, prior=_lprior)[0]
    #     return flat_samples
    #
    def calc_fiducial_profile(self, nwalkers=16, do_fit=True, **kwargs):
        self.ACP = self._calc_profile()
        prof = self.ACP.to_profile()
        container = {"prof": prof}

        if do_fit:
            self.zmean = np.mean(self.target[self.z_key])
            parmaker = make_params(z=self.zmean, cosmo=default_cosmo)
            self.params = parmaker.params

            prof = self.ACP.to_profile()
            data = get_scales(self.ACP)
            self.flat_samples = self._fit_model(data, nwalkers=nwalkers, **kwargs)
            container.update({"flat_samples": self.flat_samples})

        fname = self.file_tag + "_default_profile.p"
        print(fname)
        pickle.dump(container, open(fname, "wb"))

        # self._calc_prior(**kwargs)

    # def _calc_prior(self, factor=50., **kwargs):
    #     cov = sklearn.covariance.empirical_covariance(self.flat_samples)
    #     self.pmean = self.flat_samples.mean(axis=0)
    #     self.pcov = cov * factor
    #     self.ppp = np.random.multivariate_normal(self.flat_samples.mean(axis=0), cov=self.pcov, size=int(2e5))
    #     # self.lprior = log_prior(self.pmean, self.pcov)
    #     self.lprior = log_prior()
    #
    def calc_weights(self, score, qq, **kwargs):
        q_low = self._quintiles[qq][0]
        q_high = self._quintiles[qq][1]

        val_low = -np.inf
        if q_low != 0:
            val_low = np.percentile(score, q_low)

        val_high = np.inf
        if q_high != 100:
            val_high = np.percentile(score, q_high)

        _ww = pd.DataFrame()
        _ww[self.id_key] = self.raw_ACP.target[self.id_key]
        tmp = pd.DataFrame()

        tmp[self.id_key] = self.features[self.id_key]
        tmp["WEIGHT"] = 0.
        ii = ((score > val_low) & (score < val_high))
        tmp["WEIGHT"][ii] = 1.

        ww = pd.merge(_ww, tmp, on=self.id_key, how="left").fillna(value=0.)

        return ww

    def set_features(self, features):
        tmp = pd.merge(pd.DataFrame(self.target["MEM_MATCH_ID"]), features, on="MEM_MATCH_ID", how="left")
        self.features = tmp.dropna()
        tmp = self.features.drop(columns=["MEM_MATCH_ID",])

        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(tmp)
        self.feats = self.scaler.transform(tmp)

        self.pca = sklearn.decomposition.PCA()
        self.pca.fit(self.feats)
        self.eff = self.pca.transform(self.feats)

    def _calc_q_prof(self, feat, iq, tag, nwalkers=16, do_fit=True, **kwargs):

        print(iq)
        print(self._quintiles[iq])
        ww = self.calc_weights(feat, iq)
        # prof = self._calc_profile(weights=ww).to_profile()
        # container = {"ww": ww, "prof": prof}

        # if do_fit:
        #     zmean = np.average(self.target[self.z_key], weights=ww)
        #     print("mean-z", zmean)
        #     parmaker = make_params(z=zmean, cosmo=default_cosmo)
        #     params = parmaker.params
        #
        #     data = get_scales(prof)
        #     flat_samples = self._fit_model(data, nwalkers=nwalkers, prior=self.lprior, params=params, **kwargs)
        #     container.update({"flat_samples": flat_samples})

        # fname = self.file_tag + "_prof_"+tag+"_q"+str(iq)+".p"
        # print(fname)
        # pickle.dump(container, open(fname, "wb"))

    # def calc_ref_profiles(self):
    #     feat = self.features["LAMBDA_CHISQ"].values
    #     print("calculating reference profiles")
    #     for iq in np.arange(5):
    #         print("starting quintile ", str(iq))
    #         self._calc_q_prof(feat, iq, "ref")
    #
    # def calc_eff_profiles(self):
    #     print("calculating PCA-space split profiles")
    #     for col in np.arange(self.eff.shape[1]):
    #         print("starting eigen-feature ", str(col))
    #         feat = self.eff[:, col]
    #         for iq in np.arange(5):
    #             print("starting quintile ", str(iq), "of col", str(col))
    #             self._calc_q_prof(feat, iq, "eff-feat-"+str(col))
    #
    def calc_feat_profiles(self, do_fit=True):
        print("calculating reference profiles")
        for col in np.arange(self.feats.shape[1]):
            print("starting feature ", str(col))
            feat = self.feats[:, col]
            for iq in np.arange(5):
                print("starting quintile ", str(iq), "of col", str(col))
                self._calc_q_prof(feat, iq, "feat-"+str(col), fit=do_fit)