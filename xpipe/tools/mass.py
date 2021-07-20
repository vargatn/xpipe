
import numpy as np
import pickle
import emcee
from cluster_toolkit import deltasigma
import sklearn
from cluster_toolkit import xi
from cluster_toolkit import bias

import pandas as pd
from classy import Class
from multiprocessing import Pool

import astropy.cosmology as cosmology
from ..xhandle import pzboost
from ..xhandle import shearops

default_cosmo = cosmology.FlatLambdaCDM(Om0=0.3, H0=70)


def get_scales(prof, rmin=0.2, rmax=30):
    rr = prof.rr
    ii = (rmin <= rr) & (rr < rmax)
    data = {
        "rarr": prof.rr[ii],
        "dst": prof.dst[ii],
        "dst_err": prof.dst_err[ii]
    }

    return data


class make_params(object):
    def __init__(self, z, cosmo):
        self.cosmo = cosmo
        # Start by specifying the cosmology
        self.Omega_b = 0.05
        self.Omega_m = 0.3
        self.Omega_cdm = self.Omega_m - self.Omega_b
        self.h = 0.7 #H0/100
        self.A_s = 2.1e-9
        self.n_s = 0.96

        #Create a params dictionary
        #Need to specify the max wavenumber
        self.k_max = 10 #UNITS: 1/Mpc

        self.classparams = {
            'output':'mPk',
            'non linear':'halofit',
            'Omega_b':self.Omega_b,
            'Omega_cdm':self.Omega_cdm,
            'h':self.h,
            'A_s':self.A_s,
            'n_s':self.n_s,
            'P_k_max_1/Mpc':self.k_max,
            'z_max_pk':10. #Default value is 10
        }

        #Initialize the cosmology andcompute everything
        self.classcosmo = Class()
        self.classcosmo.set(self.classparams)
        self.classcosmo.compute()

        #Specify k and z
        self.k = np.logspace(-7, np.log10(self.k_max), num=5000) #Mpc^-1
        self.z = z

        #Call these for the nonlinear and linear matter power spectra
        self.P_nonlin = np.array([self.classcosmo.pk(ki, self.z) for ki in self.k])
        self.P_lin = np.array([self.classcosmo.pk_lin(ki, self.z) for ki in self.k])

        #NOTE: You will need to convert these to h/Mpc and (Mpc/h)^3
        #to use in the toolkit. To do this you would do:
        self.k /= self.h
        self.P_lin *= self.h**3
        self.P_nonlin *= self.h**3

        self.scale_factor = cosmo.scale_factor(self.z)

        self.params = {
            "Omega_m": self.Omega_m, # 0.3
            "h": self.h, # 0.7
            "P_lin": self.P_lin,
            "P_nonlin": self.P_nonlin,
            "k": self.k,
            "scale_factor": self.scale_factor,

        }

def calc_nfw(rarr, logmass, c, bias, params, component="both"):


    _rarr = rarr * params["h"] / params["scale_factor"]
    mass = 10**logmass * params["h"]

    #     radii = np.logspace(-3, 4, 10000) #Mpc/h comoving
    #     xi_nfw = xi.xi_nfw_at_r(radii, mass, c, params["Omega_m"])
    #     xi_mm = xi.xi_mm_at_r(radii, params["k"], params["P_lin"])
    #     xi_2halo = xi.xi_2halo(bias, xi_mm)
    #
    #     xi_hm = xi.xi_hm(xi_nfw, xi_2halo)
    #     if component == "1h":
    #         xi_hm = xi_nfw
    #         print(xi_nfw)
    #     if component == "2h":
    #         xi_hm = xi.xi_hm(xi_2halo, xi_2halo)
    #         xi_hm = xi_2halo

    R_perp = np.logspace(-3, 3, 1000) #Mpc/h comoving; distance on the sky
    Sigma = deltasigma.Sigma_nfw_at_R(R_perp, mass, c, params["Omega_m"])
    #     Sigma = deltasigma.Sigma_at_R(R_perp, radii, xi_hm, mass, c, params["Omega_m"])
    DeltaSigma = deltasigma.DeltaSigma_at_R(_rarr, R_perp, Sigma, mass, c, params["Omega_m"])

    #     factor = (1 / h) / (1 / h * scale_factor)**2
    factor = params["h"]/ params["scale_factor"]**2
    return DeltaSigma * factor


def calc_model(rarr, logmass, c, bias, params, component="both"):
    """
    Input Mpc and Msun
    Inner calculation in Mpc / h and Msun / h


    Individual conversions:
    -----------------------

    Mpc (prop) -> Mpc / h (com):
        value * h / a
        value * 0.7 / 0.75

    Msun -> Msun / h:
        value * h
        value * 0.7

    Msun / h -> Msun:
        value / h

    Mpc / h (com) -> Mpc (prop):
        value / h * a

    surface density (h + com) -> surface density (prop):
        value / h / (a / h)^2 = value / a^2 * h

        value  / 0.7 / (0.75 / 0.7)^2

    The cluster_tool function is with /h and in comoving units

    mass = 1e14 #Msun/h
    concentration = 4 #arbitrary
    """

    _rarr = rarr * params["h"] / params["scale_factor"]
    mass = 10**logmass * params["h"]

    radii = np.logspace(-3, 3, 200) #Mpc/h comoving
    xi_nfw = xi.xi_nfw_at_r(radii, mass, c, params["Omega_m"])
    xi_mm = xi.xi_mm_at_r(radii, params["k"], params["P_nonlin"])
    xi_2halo = xi.xi_2halo(bias, xi_mm)

    xi_hm = xi.xi_hm(xi_nfw, xi_2halo)
    if component == "1h":
        xi_hm = xi_nfw
    # elif component == "2h":
    #     xi_hm = xi_2halo

    # R_perp = np.logspace(-3, 2.4, 100) #Mpc/h comoving; distance on the sky
    Sigma = deltasigma.Sigma_at_R(_rarr, radii, xi_hm, mass, c, params["Omega_m"])
    DeltaSigma = deltasigma.DeltaSigma_at_R(_rarr, _rarr, Sigma, mass, c, params["Omega_m"])

    #     factor = (1 / h) / (1 / h * scale_factor)**2
    factor = params["h"]/ params["scale_factor"]**2
    return DeltaSigma * factor


def log_likelihood(theta, data, params):
    logmass = theta[0]
    c = theta[1]
    bias = theta[2]

    x = data["rarr"]
    y = data["dst"]
    yerr = data["dst_err"]

    model = calc_model(x, logmass, c, bias, params)
    sigma2 = yerr ** 2
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

def log_probability(theta, data, params, log_prior):

    lp = 0
    if log_prior is not None:
        lp = log_prior(theta)

    ll = log_likelihood(theta, data, params)
    if np. isnan(ll):
        ll = -99999.
    return lp + ll

def do_mcmc(data, params, nstep=1000, nwalkers=16, prior=None):
    #     nwalkers = 32
    ndim = 3

    pos = np.array(((14.4, 4., 2.),))
    pos = pos + 1e-2 * np.random.randn(nwalkers, 3)

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(data, params, prior), pool=pool)
        sampler.run_mcmc(pos, nstep, progress=True);
        flat_samples = sampler.get_chain(discard=500, thin=1, flat=True)

    return flat_samples, sampler


class log_prior(object):
    def __init__(self, pmean, pcov):
        self.pmean = pmean
        self.pcov = pcov
        self.BADVAL = -99999

    def __call__(self, theta):
        """Mass, concentration, bias"""
        lp = 0
        if (self.pmean is not None) and (self.pcov is not None):
            delta = theta - self.pmean

            lp = -1. * np.dot(np.dot(delta, np.linalg.inv(self.pcov)), delta.T)

        if not np.isfinite(lp):
            lp = self.BADVAL
        if np.min(theta) < 0:
            lp = self.BADVAL
        return lp

class QuintileExplorer(object):
    def __init__(self, src, flist, flist_jk, file_tag="autosplit_v1", pairs_to_load=None,
                 z_key="Z_LAMBDA", l_key="LAMBDA_CHISQ", id_key="MEM_MATCH_ID", **kwargs):
        self.src = src
        self.pair_path = pairs_to_load
        self.z_key = z_key
        self.l_key = l_key
        self.flist = flist
        self.flist_jk = flist_jk
        self.file_tag = file_tag
        self.id_key = id_key

        # self._quintiles = ((0, 20), (20, 40), (40, 60), (60, 80), (80, 100))
        self._quintiles = ((0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 100))

    def _calc_profile(self, weights=None, **kwargs):
        ACP = shearops.AutoCalibrateProfile(self.flist, self.flist_jk, self.src, weights=weights)
        ACP.get_profiles()

        smb = pzboost.SOMBoost(self.src, [self.flist_jk,], pairs_to_load=self.pair_path)
        smb.prep_boost()
        smb.get_boost(**kwargs)

        ACP.add_boost(smb)
        return ACP

    def _fit_model(self, data, nwalkers=16, **kwargs) :
        flat_samples = do_mcmc(data, self.params, nwalkers=nwalkers)[0]
        return flat_samples

    def calc_fiducial_profile(self, nwalkers=16, **kwargs):
        self.ACP = self._calc_profile()

        self.zmean = np.mean(self.ACP.target[self.z_key])
        parmaker = make_params(z=self.zmean, cosmo=default_cosmo)
        self.params = parmaker.params

        prof = self.ACP.to_profile()
        data = get_scales(self.ACP)
        self.flat_samples = self._fit_model(data, nwalkers=nwalkers, **kwargs)

        container = {"prof": prof, "flat_samples": self.flat_samples}
        fname = self.file_tag + "_default_profile.p"
        print(fname)
        pickle.dump(container, open(fname, "wb"))

        self._calc_prior(**kwargs)

    def _calc_prior(self, factor=50., **kwargs):
        cov = sklearn.covariance.empirical_covariance(self.flat_samples)
        self.pmean = self.flat_samples.mean(axis=0)
        self.pcov = cov * factor
        self.ppp = np.random.multivariate_normal(self.flat_samples.mean(axis=0), cov=self.pcov, size=int(2e5))
        self.lprior = log_prior(self.pmean, self.pcov)

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
        _ww[self.id_key] = self.ACP.target[self.id_key]

        tmp = pd.DataFrame()

        tmp[self.id_key] = self.features[self.id_key]
        tmp["WEIGHT"] = 0.
        ii = ((score > val_low) & (score < val_high))
        tmp["WEIGHT"][ii] = 1.

        ww = pd.merge(_ww, tmp, on=self.id_key, how="left").fillna(value=0.)

        return ww

    def set_features(self, features):
        self.features = features
        tmp = self.features.drop(columns=["MEM_MATCH_ID",])

        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(tmp)
        feats = self.scaler.transform(tmp)

        self.pca = sklearn.decomposition.PCA()
        self.pca.fit(feats)
        self.eff = self.pca.transform(feats)

    def _calc_q_prof(self, feat, iq, tag, nwalkers=16, **kwargs):

        print(iq)
        print(self._quintiles[iq])
        ww = self.calc_weights(feat, iq)
        prof = self._calc_profile(weights=ww).to_profile()

        data = get_scales(prof)
        prior_flat_samples = self._fit_model(data, nwalkers=nwalkers, prior=self.lprior, **kwargs)

        container = {"ww": ww, "prof": prof, "flat_samples": prior_flat_samples}
        fname = self.file_tag + "_prof_"+tag+"_q"+str(iq)+".p"
        print(fname)
        pickle.dump(container, open(fname, "wb"))

    def calc_ref_profiles(self):
        feat = self.features["LAMBDA_CHISQ"].values
        print("calculating reference profiles")
        for iq in np.arange(10):
            print("starting decile ", str(iq))
            self._calc_q_prof(feat, iq, "ref")

    def calc_eff_profiles(self):
        print("calculating reference profiles")
        for col in np.arange(self.eff.shape[1]):
            print("starting eigen-feature ", str(col))
            feat = self.eff[:, col]
            for iq in np.arange(10):
                print("starting decile ", str(iq), "of col", str(col))
                self._calc_q_prof(feat, iq, "feat-"+str(col))







