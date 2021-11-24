
import numpy as np
import emcee
from cluster_toolkit import deltasigma
from cluster_toolkit import xi
from cluster_toolkit import miscentering
from cluster_toolkit import bias
from classy import Class
from multiprocessing import Pool

import astropy.cosmology as cosmology

default_cosmo = cosmology.FlatLambdaCDM(Om0=0.3, H0=70)


def get_scales(prof, rmin=0.1, rmax=100):
    rr = prof.rr
    ii = (rmin <= rr) & (rr < rmax)
    data = {
        "rarr": prof.rr[ii],
        "dst": prof.dst[ii],
        "dst_err": prof.dst_err[ii]
    }

    return data


class make_params(object):
    def __init__(self, z, cosmo, logmass_array=np.linspace(12, 16, 1000)):
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

        self.mass_array = 10**logmass_array * self.h
        self.bias_array = bias.bias_at_M(self.mass_array, self.k, self.P_lin, self.Omega_m)

        self.params = {
            "Omega_m": self.Omega_m, # 0.3
            "h": self.h, # 0.7
            "P_lin": self.P_lin,
            "P_nonlin": self.P_nonlin,
            "k": self.k,
            "scale_factor": self.scale_factor,
            "logmass_array": logmass_array,
            "bias_array": self.bias_array,

        }

def calc_nfw(rarr, logmass, c, params):

    _rarr = rarr * params["h"] / params["scale_factor"]
    mass = 10**logmass * params["h"]

    R_perp = np.logspace(-3, 3, 1000) #Mpc/h comoving; distance on the sky
    Sigma = deltasigma.Sigma_nfw_at_R(R_perp, mass, c, params["Omega_m"])
    #     Sigma = deltasigma.Sigma_at_R(R_perp, radii, xi_hm, mass, c, params["Omega_m"])
    DeltaSigma = deltasigma.DeltaSigma_at_R(_rarr, R_perp, Sigma, mass, c, params["Omega_m"])

    #     factor = (1 / h) / (1 / h * scale_factor)**2
    factor = params["h"]/ params["scale_factor"]**2
    return DeltaSigma * factor

def calc_single_misc_nfw(rarr, logmass, c, R_misc, params):
    _rarr = rarr * params["h"] / params["scale_factor"]
    _R_misc = R_misc * params["h"] / params["scale_factor"]
    mass = 10**logmass * params["h"]

    R_perp = np.logspace(-3, 3, 1000) #Mpc/h comoving; distance on the sky
    Sigma = deltasigma.Sigma_nfw_at_R(R_perp, mass, c, params["Omega_m"])
    Sigma_misc = miscentering.Sigma_mis_single_at_R(R_perp, R_perp, Sigma, mass, c, params["Omega_m"], _R_misc)
    DeltaSigma = miscentering.DeltaSigma_mis_at_R(_rarr, R_perp, Sigma_misc)

    factor = params["h"]/ params["scale_factor"]**2
    return DeltaSigma * factor


def calc_misc_nfw(rarr, logmass, c, R_misc, params):
    _rarr = rarr * params["h"] / params["scale_factor"]
    _R_misc = R_misc * params["h"] / params["scale_factor"]
    mass = 10**logmass * params["h"]

    R_perp = np.logspace(-3, 3, 1000) #Mpc/h comoving; distance on the sky
    Sigma = deltasigma.Sigma_nfw_at_R(R_perp, mass, c, params["Omega_m"])
    Sigma_misc = miscentering.Sigma_mis_at_R(R_perp, R_perp, Sigma, mass, c, params["Omega_m"], _R_misc)
    DeltaSigma = miscentering.DeltaSigma_mis_at_R(_rarr, R_perp, Sigma_misc)

    factor = params["h"]/ params["scale_factor"]**2
    return DeltaSigma * factor

def calc_misc_model(rarr, logmass, c, bias, R_misc, params):
    _rarr = rarr * params["h"] / params["scale_factor"]
    _R_misc = R_misc * params["h"] / params["scale_factor"]
    mass = 10**logmass * params["h"]

    radii = np.logspace(-3, 3, 400) #Mpc/h comoving
    xi_nfw = xi.xi_nfw_at_r(radii, mass, c, params["Omega_m"])
    xi_mm = xi.xi_mm_at_r(radii, params["k"], params["P_nonlin"])
    xi_2halo = xi.xi_2halo(bias, xi_mm)

    xi_hm = xi.xi_hm(xi_nfw, xi_2halo)
    R_perp = np.logspace(-3, 2.4, 300) #Mpc/h comoving; distance on the sky
    Sigma = deltasigma.Sigma_at_R(R_perp, radii, xi_hm, mass, c, params["Omega_m"])
    Sigma_misc = miscentering.Sigma_mis_at_R(R_perp, R_perp, Sigma, mass, c, params["Omega_m"], _R_misc)
    DeltaSigma = miscentering.DeltaSigma_mis_at_R(_rarr, R_perp, Sigma_misc)

    factor = params["h"]/ params["scale_factor"]**2
    return DeltaSigma * factor

def calc_sub_misc_nfw(rarr, logmass, c, R_misc, f_cen, dist, params):
    _rarr = rarr * params["h"] / params["scale_factor"]
    _R_misc = R_misc * params["h"] / params["scale_factor"]
    _dist = dist * params["h"] / params["scale_factor"]
    mass = 10**logmass * params["h"]

    R_perp = np.logspace(-3, 2.4, 1000) #Mpc/h comoving; distance on the sky

    Sigma = deltasigma.Sigma_nfw_at_R(R_perp, mass, c, params["Omega_m"])
    Sigma_misc = miscentering.Sigma_mis_at_R(R_perp, R_perp, Sigma, mass, c, params["Omega_m"], _R_misc)
    Sigma_clust = f_cen * Sigma + (1 - f_cen) * Sigma_misc

    Sigma_sub_clust = miscentering.Sigma_mis_single_at_R(R_perp, R_perp, Sigma_clust, mass, c, params["Omega_m"], _dist)
    DeltaSigma = miscentering.DeltaSigma_mis_at_R(_rarr, R_perp, Sigma_sub_clust)

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

    radii = np.logspace(-3, 3, 400) #Mpc/h comoving
    xi_nfw = xi.xi_nfw_at_r(radii, mass, c, params["Omega_m"])
    xi_mm = xi.xi_mm_at_r(radii, params["k"], params["P_nonlin"])
    xi_2halo = xi.xi_2halo(bias, xi_mm)

    xi_hm = xi.xi_hm(xi_nfw, xi_2halo)
    if component == "1h":
        xi_hm = xi_nfw

    R_perp = np.logspace(-3, 2.4, 300) #Mpc/h comoving; distance on the sky
    Sigma = deltasigma.Sigma_at_R(R_perp, radii, xi_hm, mass, c, params["Omega_m"])
    # return Sigma
    DeltaSigma = deltasigma.DeltaSigma_at_R(_rarr, R_perp, Sigma, mass, c, params["Omega_m"])
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
        sampler.run_mcmc(pos, nstep, progress=True)
        flat_samples = sampler.get_chain(discard=500, thin=1, flat=True)

    return flat_samples, sampler



class log_prior(object):
    def __init__(self, pmean=None, pcov=None):
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

class log_cluster_prior(object):
    def __init__(self, logmass_edges=(11, 18), c_edges=(0, 20), tau_pars=(0.17, 0.04), f_pars=(0.75, 0.08)):
        """logmass, c linear edges, tau, f gaussian prior (and zero cut)"""
        self.logmass_edges = logmass_edges
        self.c_edges = c_edges
        self.tau_pars = tau_pars
        self.f_pars = f_pars
        self.BADVAL = -99999

    def __call__(self, theta):
        """logmass, c, tau, f_cen"""
        logmass, c, tau, f_cen = theta
        lp = 0

        if (self.logmass_edges[0] > logmass) or (self.logmass_edges[1] < logmass):
            lp = self.BADVAL
        if (self.c_edges[0] > c) or (self.c_edges[1] < c):
            lp = self.BADVAL

        lp += -np.log(self.tau_pars[1]) - (tau - self.tau_pars[0])**2 / (2 * self.tau_pars[1]**2)
        lp += -np.log(self.f_pars[1]) - (f_cen - self.f_pars[0])**2 / (2 * self.f_pars[1]**2)

        if not np.isfinite(lp):
            lp = self.BADVAL
        if np.min(theta) < 0:
            lp = self.BADVAL
        return lp


def calc_bias(logmass, params):
    mass = 10**logmass * params["h"]

    b = bias.bias_at_M(mass, params["k"], params["P_lin"], params["Omega_m"])
    return b

def calc_cluster_model(x, logmass, c, R_misc, f_cen, params):
    """Assuming Gamma miscentering kernel"""

    # b = calc_bias(logmass, params)
    b = params["bias_array"][np.argmin((logmass - params["logmass_array"])**2)]

    model_cen = calc_model(x, logmass, c, b, params)
    model_miscen = calc_misc_model(x, logmass, c, b, R_misc, params)

    model = f_cen * model_cen + (1 - f_cen) * model_miscen
    return model

def log_cluster_likelihood(theta, data, params):
    logmass = theta[0]
    c = theta[1]
    tau_misc = theta[2]
    f_cen = theta[3]

    x = data["rarr"]
    y = data["dst"]
    cov = data["cov"]
    R_lambda = data["R_lambda"]
    R_misc = tau_misc * R_lambda


    model = calc_cluster_model(x, logmass, c, R_misc, f_cen, params)

    dvec = y - model
    lprop = -0.5 * np.dot(np.dot(dvec.T, np.linalg.inv(cov)), dvec)
    return lprop

def log_cluster_probability(theta, data, params, log_prior):

    lp = 0
    if log_prior is not None:
        lp = log_prior(theta)

    ll = log_cluster_likelihood(theta, data, params)
    if np. isnan(ll):
        ll = -99999.
    return lp + ll

def do_cluster_mcmc(data, params, nstep=1000, nwalkers=16, prior=None, init_pos=np.array([14.4, 4., 0.4, 0.8]), init_fac=1e-2, discard=200):
    ndim = len(init_pos)
    pos = init_pos[:, np.newaxis].T + init_fac * np.random.randn(nwalkers, ndim)

    try:
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_cluster_probability, args=(data, params, prior), pool=pool)
            sampler.run_mcmc(pos, nstep, progress=True)
            flat_samples = sampler.get_chain(discard=discard, thin=1, flat=True)
    except KeyboardInterrupt:
        raise KeyboardInterrupt

    return flat_samples, sampler








