import pickle
import numpy as np
import pandas as pd
import sklearn
import os

from xpipe.likelihood.mass import make_params, default_cosmo
from xpipe.xhandle import shearops, pzboost

from .mcmc import log_cluster_prob, do_mcmc

# TODO add hartlap correction
# TODO add shear bias scatter
# TODO add photo-z bias scatter


CLUST_RADIAL_EDGES = np.logspace(np.log10(0.1), np.log10(100), 16)
CLUST_RADIAL_DENSE = np.logspace(np.log10(0.02), 2, 100)

def get_scales(prof, rmin=0.1, rmax=100, diag_cov=None):
    """

    Parameters
    ----------
    prof: AutoCorrectedProfile
        profile to processs
    rmin: float
        minimum radius
    rmax: float
        maximum radius
    diag_cov: np.array
        diagonal elements of the cov to be added in quadrature

    Returns
    -------
    data: dict
        data container cut to scale

    """
    rr = prof.rr
    ii = (rmin <= rr) & (rr < rmax)

    err = prof.dst_err[ii]
    cov = prof.cov[ii, :][:, ii],
    if diag_cov is not None:
        cov += diag_cov[ii]
        err += err + np.sqrt(diag_cov[ii])
    data = {
        "rarr": prof.rr[ii],
        "dst": prof.dst[ii],
        "dst_err": err,
        "cov": cov,
        "rbin_edges": CLUST_RADIAL_EDGES,
        "rarr_dense": CLUST_RADIAL_DENSE,
    }

    return data


# Cluster likelihood is mcmc.log_cluster_prob()
class QuintileExplorer(object):
    def __init__(self, src, flist, flist_jk, file_tag="autosplit_v1", pairs_to_load=None,
                 z_key="Z_LAMBDA", l_key="LAMBDA_CHISQ", id_key="MEM_MATCH_ID",
                 ismeta=False, bins_to_use=np.linspace(0, 14, 15), npdf=15, init_pos=(14.3,  4.5, 0.15,  0.83),
                 nstep=1000, nwalkers=16, init_fac=1e-2, discard=200, R_lambda=0.88, scinv=0.0003, scales=(0.1, 100),
                 Rs_sbins=None, ms_sbins=None, ms_stds=None, **kwargs):
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

        self.init_pos = init_pos
        self.nstep = nstep
        self.nwalkers=nwalkers
        self.init_fac = init_fac
        self.discard = discard
        self.R_lambda = R_lambda
        self.scinv = scinv
        self.scales = scales
        self.Rs_sbins = Rs_sbins
        self.ms_sbins = ms_sbins
        self.ms_stds = ms_stds

        self.lprior = None

        self._quintiles = ((0, 20), (20, 40), (40, 60), (60, 80), (80, 100))

    def load_target(self):
        self.raw_ACP = shearops.AutoCalibrateProfile(self.flist, self.flist_jk, self.src, xlims=(0.1, 100), Rs_sbins=self.Rs_sbins)
        self.raw_ACP.get_profiles(ismeta=self.ismeta, Rs_sbins=self.Rs_sbins, mfactor_sbins=self.ms_sbins)
        self.target = self.raw_ACP.target
        self.smb = pzboost.SOMBoost(self.src, [self.flist_jk,], pairs_to_load=self.pair_path)

    def _calc_profile(self, weights=None, _include_boost=True, **kwargs):
        ACP = self.raw_ACP.copy()
        ACP.get_profiles(reload=False, ismeta=self.ismeta, weights=weights, Rs_sbins=self.Rs_sbins, mfactor_sbins=self.ms_sbins)

        if _include_boost:
            self.smb.prep_boost(bins_to_use=np.linspace(0, 14, 15))
            self.smb.get_boost_jk(npdf=15, **kwargs)
            ACP.add_boost_jk(self.smb, mfactor_sbins=self.ms_sbins)

        return ACP

    def _fit_model(self, data, params=None, **kwargs):

        if params is None:
            params = self.params

        lcp = log_cluster_prob(data, params, use_rcens=True)
        flat_samples, sampler = do_mcmc(lcp, self.init_pos, self.nstep, self.nwalkers, self.init_fac, self.discard)
        return flat_samples, sampler

    def get_params(self):
        self.zmean = np.mean(self.target[self.z_key])
        parmaker = make_params(z=self.zmean, cosmo=default_cosmo)
        parmaker.params.update({"scinv": self.scinv})
        self.params = parmaker

    def calc_fiducial_profile(self, do_fit=True, _include_boost=True, save=True, **kwargs):
        self.ACP = self._calc_profile(_include_boost=_include_boost)
        prof = self.ACP.to_profile()
        container = {"prof": prof}

        if do_fit:
            self.get_params()
            self.R_lambda = np.average(self.target["R_LAMBDA"])
            data = get_scales(self.ACP, rmin=self.scales[0], rmax=self.scales[1])
            print(data["dst"])
            data.update({"R_lambda": self.R_lambda})
            self.flat_samples, self.sampler = self._fit_model(data, **kwargs)
            container.update({"flat_samples": self.flat_samples, "sampler": self.sampler, "params": self.params.params})
            container.update({"scinv": self.scinv, "z": self.zmean, "R_lambda": self.R_lambda})

        if save:
            fname = self.file_tag + "_default_profile.p"
            print(fname)
            pickle.dump(container, open(fname, "wb"))

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

        print(score.min(), val_low)
        print(score.max(), val_high)
        _ww = pd.DataFrame()
        _ww[self.id_key] = self.raw_ACP.target[self.id_key]
        tmp = pd.DataFrame()

        tmp[self.id_key] = self.features[self.id_key]
        tmp["WEIGHT"] = 0.
        ii = ((score > val_low) & (score <= val_high))
        tmp["WEIGHT"][ii] = 1.

        ww = pd.merge(_ww, tmp, on=self.id_key, how="left").fillna(value=0.)
        print("weight fraction", str(ww["WEIGHT"].sum() / len(ww)))
        return ww

    def calc_weights_dual(self, score1, score2, qq, **kwargs):
        """uses the mean of two feature percentiles to assign quintile weights"""
        _ww = pd.DataFrame()
        _ww[self.id_key] = self.raw_ACP.target[self.id_key]
        tmp = pd.DataFrame()

        pvals1 = np.array([np.percentile(score1, p) for p in np.arange(101)])
        ppvals1 = np.zeros(shape=score1.shape) - 9999
        for i in np.arange(100):
            ii = np.where((score1 >= pvals1[i]) & (score1 < pvals1[i + 1]))
            ppvals1[ii] = i
        ii = np.where(score1 == pvals1[-1])
        ppvals1[ii] = 100
        pvals2 = np.array([np.percentile(score2, p) for p in np.arange(101)])
        ppvals2 = np.zeros(shape=score2.shape) - 9999
        for i in np.arange(100):
            ii = np.where((score2 >= pvals2[i]) & (score2 < pvals2[i + 1]))
            ppvals2[ii] = i
        ii = np.where(score2 == pvals2[-1])
        ppvals2[ii] = 100

        score = np.vstack((ppvals1, ppvals2)).T.mean(axis=1)
        # print(pvals1)
        # print(pvals2)
        # print(score)
        q_low = self._quintiles[qq][0]
        q_high = self._quintiles[qq][1]

        val_low = -np.inf
        if q_low != 0:
            val_low = np.percentile(score, q_low)

        val_high = np.inf
        if q_high != 100:
            val_high = np.percentile(score, q_high)

        print(score.min(), val_low)
        print(score.max(), val_high)
        _ww = pd.DataFrame()
        _ww[self.id_key] = self.raw_ACP.target[self.id_key]
        tmp = pd.DataFrame()

        tmp[self.id_key] = self.features[self.id_key]
        tmp["WEIGHT"] = 0.
        ii = ((score > val_low) & (score <= val_high))
        tmp["WEIGHT"][ii] = 1.

        ww = pd.merge(_ww, tmp, on=self.id_key, how="left").fillna(value=0.)
        print("weight fraction", str(ww["WEIGHT"].sum() / len(ww)))
        return ww#, tmp

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

    def _calc_q_prof(self, feat, iq, tag, nwalkers=16, do_fit=True, _include_boost=True, feat2=None, overwrite=True, **kwargs):
        fname = self.file_tag + "_prof_"+tag+"_q"+str(iq)+".p"
        print(fname)
        if os.path.isfile(fname) and (not overwrite):
            print("skipping file...")
            return None
        else:
            print("starting profile calculation")

        # print(iq)
        # print(self._quintiles[iq])

        if feat2 is not None:
            ww = self.calc_weights_dual(feat, feat2, iq)
        else:
            ww = self.calc_weights(feat, iq)

        if not ww["WEIGHT"].sum():
            container = {"ww": ww, "prof": None}
        else:
            prof = self._calc_profile(weights=ww, _include_boost=_include_boost).to_profile()
            container = {"ww": ww, "prof": prof}

            if do_fit:
                zmean = np.average(self.target[self.z_key], weights=ww["WEIGHT"])
                print("mean-z", zmean)
                parmaker = make_params(z=zmean, cosmo=default_cosmo)
                parmaker.params.update({"scinv": self.scinv})
                self.params = parmaker

                r_lambda = np.average(self.target["R_LAMBDA"], weights=ww["WEIGHT"])

                data = get_scales(prof, rmin=self.scales[0], rmax=self.scales[1])
                data.update({"R_lambda": r_lambda})
                self.flat_samples, self.sampler = self._fit_model(data, nwalkers=nwalkers, prior=self.lprior,
                                                                  params=parmaker, **kwargs)
                container.update({"flat_samples": self.flat_samples, "sampler": self.sampler, "params": self.params.params})
                container.update({"scinv": self.scinv, "z": zmean, "R_lambda": self.R_lambda})

        print(fname)
        pickle.dump(container, open(fname, "wb"))

    def calc_ref_profiles(self, do_fit=True, _include_boost=True, overwrite=True):
        feat = self.features["LAMBDA_CHISQ"].values
        print("calculating reference profiles")
        for iq in np.arange(len(self._quintiles)):
            print("starting quintile ", str(iq))
            self._calc_q_prof(feat, iq, "ref", do_fit=do_fit, _include_boost=_include_boost, overwrite=overwrite)

    def calc_custom_profiles(self, feat, do_fit=True, _include_boost=True, tag="custom", overwrite=True):
        print("calculating reference profiles")
        for iq in np.arange(len(self._quintiles)):
            print("starting quintile ", str(iq))
            self._calc_q_prof(feat, iq, tag, do_fit=do_fit, _include_boost=_include_boost, overwrite=overwrite)

    def calc_eff_profiles(self, do_fit=True, _include_boost=True, overwrite=True):
        print("calculating PCA-space split profiles")
        for col in np.arange(self.eff.shape[1]):
            print("starting eigen-feature ", str(col))
            feat = self.eff[:, col]
            for iq in np.arange(len(self._quintiles)):
                print("starting quintile ", str(iq), "of col", str(col))
                self._calc_q_prof(feat, iq, "eff-feat-"+str(col), do_fit=do_fit, _include_boost=_include_boost, overwrite=overwrite)

    def calc_feat_profiles(self, do_fit=True, _include_boost=True, overwrite=True):
        print("calculating reference profiles")
        for col in np.arange(self.feats.shape[1]):
            print("starting feature ", str(col))
            feat = self.feats[:, col]
            for iq in np.arange(len(self._quintiles)):
                print("starting quintile ", str(iq), "of col", str(col))
                self._calc_q_prof(feat, iq, "feat-"+str(col), do_fit=do_fit, _include_boost=_include_boost, overwrite=overwrite)

    def calc_custom_expand_profiles(self, feat1, feat2, do_fit=True, _include_boost=True, tag="custom_expand", overwrite=True):
        print("calculating 1-feature expansion profiles")
        for iq in np.arange(len(self._quintiles)):
            print("starting quintile ", str(iq))
            self._calc_q_prof(feat1, iq, tag, do_fit=do_fit, _include_boost=_include_boost, feat2=feat2, overwrite=overwrite)


def calc_weights_dual(score1, score2, qq, quintiles=((0, 20), (20, 40), (40, 60), (60, 80), (80, 100)),**kwargs):
    """uses the mean of two feature percentiles to assign quintile weights"""
    pvals1 = np.array([np.percentile(score1, p) for p in np.arange(101)])
    ppvals1 = np.zeros(shape=score1.shape) - 9999
    for i in np.arange(100):
        ii = np.where((score1 >= pvals1[i]) & (score1 < pvals1[i + 1]))
        ppvals1[ii] = i
    ii = np.where(score1 == pvals1[-1])
    ppvals1[ii] = 100
    pvals2 = np.array([np.percentile(score2, p) for p in np.arange(101)])
    ppvals2 = np.zeros(shape=score2.shape) - 9999
    for i in np.arange(100):
        ii = np.where((score2 >= pvals2[i]) & (score2 < pvals2[i + 1]))
        ppvals2[ii] = i
    ii = np.where(score2 == pvals2[-1])
    ppvals2[ii] = 100

    score = np.vstack((ppvals1, ppvals2)).T.mean(axis=1)

    q_low = quintiles[qq][0]
    q_high = quintiles[qq][1]

    val_low = -np.inf
    if q_low != 0:
        val_low = np.percentile(score, q_low)

    val_high = np.inf
    if q_high != 100:
        val_high = np.percentile(score, q_high)

    tmp = np.zeros(len(score1))
    ii = ((score > val_low) & (score <= val_high))
    tmp[ii] = 1.

    return tmp