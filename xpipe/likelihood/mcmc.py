
import numpy as np
from multiprocessing import Pool
import emcee
import cluster_toolkit.averaging as averaging

import xpipe.likelihood.mass as mass

BADVAL = -99999

models_available = {
    "cluster": mass.calc_model,
    "cluster_mixture": mass.calc_cluster_mixture_model,
    "single_sub_clust": mass.calc_single_sub_clust_model2
}





def do_mcmc(log_prob, init_pos, nstep=1000, nwalkers=16, init_fac=1e-2, discard=200):
    ndim = len(init_pos)
    pos = np.array(init_pos)[:, np.newaxis].T + init_fac * np.random.randn(nwalkers, ndim)

    with Pool(nwalkers) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, pool=pool)
        sampler.run_mcmc(pos, nstep, progress=True)
        flat_samples = sampler.get_chain(discard=discard, thin=1, flat=True)

    return flat_samples, sampler


class log_sub_cluster_free_prob(object):
    def __init__(self, data, params, nbins=5, logmass_edges=(10, 18), c_edges=(2, 20), c_sub_edges=(5, 20), tau_pars=(0.17, 0.04), f_pars=(0.75, 0.08)):
        """logmass, c linear edges, tau, f gaussian prior (and zero cut)"""
        self.data = data
        self.params = params.params
        self.logmass_edges = logmass_edges
        self.c_edges = c_edges
        self.c_sub_edges = c_sub_edges
        self.tau_pars = tau_pars
        self.f_pars = f_pars
        self.BADVAL = BADVAL
        self.nbins = nbins

    def __call__(self, theta):
        """logmass, c, tau, f_cen"""
        logmass1, c1, logmass2, c2, tau_misc, f_cen = theta
        lp = 0
        cond = (((self.logmass_edges[0] > logmass1) or (self.logmass_edges[1] < logmass1)) or
                ((self.logmass_edges[0] > logmass2) or (self.logmass_edges[1] < logmass2)) or
                ((self.c_edges[0] > c1) or (self.c_edges[1] < c1)) or
                ((self.c_edges[0] > c2) or (self.c_edges[1] < c2)) or
                ((f_cen > 1) or f_cen < 0)
                # ((self.logmass_edges[0] > logmass_bcg) or (self.logmass_edges[1] < logmass_bcg)) or
                # (10**logmass_bcg > 0.1*10**logmass1) or
                )
        if cond:
            # print("badval")
            lp = self.BADVAL
        else:
            y = self.data["dst"]
            cov = self.data["cov"]

            model = self.calc_model(theta)[1]
            dvec = y - model
            lp += -0.5 * np.dot(np.dot(dvec.T, np.linalg.inv(cov)), dvec)
            # print(lp)
            lp += -np.log(self.tau_pars[1]) - (tau_misc - self.tau_pars[0])**2 / (2 * self.tau_pars[1]**2)
            lp += -np.log(self.f_pars[1]) - (f_cen - self.f_pars[0])**2 / (2 * self.f_pars[1]**2)
            # print(lp)
        if not np.isfinite(lp):
            lp = self.BADVAL
        if np.min(theta) < 0:
            lp = self.BADVAL
        return lp

    def calc_model(self, theta):
        logmass1, c1, logmass2, c2, tau_misc, f_cen = theta
        R_lambda = self.data["R_lambda"]
        R_misc = tau_misc * R_lambda

        bb = mass.calc_bias(logmass1, self.params)
        model1 = mass.calc_cluster_mix_nw(self.data["rarr_dense"][0], logmass1, c1, bb, R_misc, f_cen, self.params)
        av_model1 = averaging.average_profile_in_bins(self.data["rbin_edges"][0], self.data["rarr_dense"][0], model1)

        # c2 = mass.get_conc(logmass2, self.params)
        model2 = mass.calc_sub_mixture_nw(self.data["rarr_dense"][1], logmass1, c1, logmass2, c2, R_misc, f_cen, self.data["dists"], self.params, nbins=self.nbins)
        av_model2 = averaging.average_profile_in_bins(self.data["rbin_edges"][1], self.data["rarr_dense"][0], model2)

        model = np.concatenate((av_model1, av_model2))
        return (model1, model2), model



class log_sub_cluster_ml_prob(object):
    def __init__(self, data, params, nbins=5, logmass_edges=(10, 18), c_edges=(2, 20), c_sub_edges=(5, 20), tau_pars=(0.17, 0.04), f_pars=(0.75, 0.08)):
        """logmass, c linear edges, tau, f gaussian prior (and zero cut)"""
        self.data = data
        self.params = params.params
        self.logmass_edges = logmass_edges
        self.c_edges = c_edges
        self.c_sub_edges = c_sub_edges
        self.tau_pars = tau_pars
        self.f_pars = f_pars
        self.BADVAL = BADVAL
        self.nbins = nbins

    def __call__(self, theta):
        """logmass, c, tau, f_cen"""
        logmass1, c1, ml, tau_misc, f_cen = theta
        lp = 0
        cond = (((self.logmass_edges[0] > logmass1) or (self.logmass_edges[1] < logmass1)) or
                ((self.c_edges[0] > c1) or (self.c_edges[1] < c1)) or
                ((ml < 1e-3) or (ml > 1e4)) or
                ((f_cen > 1) or f_cen < 0)
                # ((self.logmass_edges[0] > logmass_bcg) or (self.logmass_edges[1] < logmass_bcg)) or
                # (10**logmass_bcg > 0.1*10**logmass1) or
                )
        if cond:
            # print("badval")
            lp = self.BADVAL
        else:
            y = self.data["dst"]
            cov = self.data["cov"]
            # print(y.shape)

            model = self.calc_model(theta)[1]
            # print(model.shape)
            dvec = y - model
            lp += -0.5 * np.dot(np.dot(dvec.T, np.linalg.inv(cov)), dvec)
            # print(lp)
            lp += -np.log(self.tau_pars[1]) - (tau_misc - self.tau_pars[0])**2 / (2 * self.tau_pars[1]**2)
            lp += -np.log(self.f_pars[1]) - (f_cen - self.f_pars[0])**2 / (2 * self.f_pars[1]**2)
            # print(lp)
        if not np.isfinite(lp):
            lp = self.BADVAL
        if np.min(theta) < 0:
            lp = self.BADVAL
        return lp

    def calc_mass(self, ml):
        lums = self.data["lsuns"]
        logmass = np.log10(ml * lums).mean()
        return logmass

    def calc_model(self, theta):
        logmass1, c1, ml, tau_misc, f_cen = theta
        R_lambda = self.data["R_lambda"]
        R_misc = tau_misc * R_lambda

        bb = mass.calc_bias(logmass1, self.params)
        model1 = mass.calc_cluster_mix_nw(self.data["rarr_dense"][0], logmass1, c1, bb, R_misc, f_cen, self.params)
        av_model1 = averaging.average_profile_in_bins(self.data["rbin_edges"][0], self.data["rarr_dense"][0], model1)

        logmass2 = self.calc_mass(ml)
        c2 = mass.get_conc(logmass2, self.params)
        model2 = mass.calc_sub_mixture_nw(self.data["rarr_dense"][1], logmass1, c1, logmass2, c2, R_misc, f_cen, self.data["dists"], self.params, nbins=self.nbins)
        av_model2 = averaging.average_profile_in_bins(self.data["rbin_edges"][1], self.data["rarr_dense"][0], model2)

        model = np.concatenate((av_model1, av_model2))
        return (model1, model2), model


class log_sub_cluster_ml_all_prob(object):
    def __init__(self, data, params, nbins=5, logmass_edges=(10, 18), c_edges=(2, 20), c_sub_edges=(5, 20), tau_pars=(0.17, 0.04), f_pars=(0.75, 0.08)):
        """logmass, c linear edges, tau, f gaussian prior (and zero cut)"""
        self.data = data
        self.params = params.params
        self.logmass_edges = logmass_edges
        self.c_edges = c_edges
        self.c_sub_edges = c_sub_edges
        self.tau_pars = tau_pars
        self.f_pars = f_pars
        self.BADVAL = BADVAL
        self.nbins = nbins

    def __call__(self, theta):
        """logmass, c, tau, f_cen"""
        logmass1, c1, ml, tau_misc, f_cen = theta
        lp = 0
        cond = (((self.logmass_edges[0] > logmass1) or (self.logmass_edges[1] < logmass1)) or
                ((self.c_edges[0] > c1) or (self.c_edges[1] < c1)) or
                ((ml < 1e-3) or (ml > 1e4)) or
                ((f_cen > 1) or f_cen < 0)
                # ((self.logmass_edges[0] > logmass_bcg) or (self.logmass_edges[1] < logmass_bcg)) or
                # (10**logmass_bcg > 0.1*10**logmass1) or
                )
        if cond:
            # print("badval")
            lp = self.BADVAL
        else:
            y = self.data["dst"]
            cov = self.data["cov"]
            # print(y.shape)

            model = self.calc_model(theta)[1]
            # print(model.shape)
            dvec = y - model
            lp += -0.5 * np.dot(np.dot(dvec.T, np.linalg.inv(cov)), dvec)
            # print(lp)
            lp += -np.log(self.tau_pars[1]) - (tau_misc - self.tau_pars[0])**2 / (2 * self.tau_pars[1]**2)
            lp += -np.log(self.f_pars[1]) - (f_cen - self.f_pars[0])**2 / (2 * self.f_pars[1]**2)
            # print(lp)
        if not np.isfinite(lp):
            lp = self.BADVAL
        # if np.min(theta) < 0:
        #     lp = self.BADVAL
        return lp

    def calc_mass(self, ml, rbin=0):
        lums = self.data["lsuns"][rbin]
        # rpivot = self.data["rpivot"]
        # dists = self.data["dists"][rbin]
        # logmass = np.log10(ml * lums * (dists / rpivot)**ralpha).mean()
        logmass = np.log10(ml * lums).mean()
        return logmass

    def calc_model(self, theta):
        logmass1, c1, ml, tau_misc, f_cen = theta
        R_lambda = self.data["R_lambda"]
        R_misc = tau_misc * R_lambda

        bb = mass.calc_bias(logmass1, self.params)
        model1 = mass.calc_cluster_mix_nw(self.data["rarr_dense"][0], logmass1, c1, bb, R_misc, f_cen, self.params)
        av_model1 = averaging.average_profile_in_bins(self.data["rbin_edges"][0], self.data["rarr_dense"][0], model1)
        # print(model1)
        logmass2 = self.calc_mass(ml, rbin=0)
        c2 = mass.get_conc(logmass2, self.params)
        # print(self.data["rarr_dense"][1])
        model2 = mass.calc_sub_mixture_nw(self.data["rarr_dense"][1], logmass1, c1, logmass2, c2, R_misc, f_cen, self.data["dists"][0], self.params, nbins=self.nbins)
        av_model2 = averaging.average_profile_in_bins(self.data["rbin_edges"][1], self.data["rarr_dense"][1], model2)
        # print(model2)
        logmass3 = self.calc_mass(ml, rbin=1)
        c3 = mass.get_conc(logmass3, self.params)
        model3 = mass.calc_sub_mixture_nw(self.data["rarr_dense"][2], logmass1, c1, logmass3, c3, R_misc, f_cen, self.data["dists"][1], self.params, nbins=self.nbins)
        av_model3 = averaging.average_profile_in_bins(self.data["rbin_edges"][2], self.data["rarr_dense"][2], model3)

        logmass4 = self.calc_mass(ml, rbin=2)
        c4 = mass.get_conc(logmass4, self.params)
        model4 = mass.calc_sub_mixture_nw(self.data["rarr_dense"][3], logmass1, c1, logmass4, c4, R_misc, f_cen, self.data["dists"][2], self.params, nbins=self.nbins)
        av_model4 = averaging.average_profile_in_bins(self.data["rbin_edges"][3], self.data["rarr_dense"][3], model4)

        logmass5 = self.calc_mass(ml, rbin=3)
        c5 = mass.get_conc(logmass5, self.params)
        model5 = mass.calc_sub_mixture_nw(self.data["rarr_dense"][4], logmass1, c1, logmass5, c5, R_misc, f_cen, self.data["dists"][3], self.params, nbins=self.nbins)
        av_model5 = averaging.average_profile_in_bins(self.data["rbin_edges"][4], self.data["rarr_dense"][4], model5)
        # model = np.concatenate((av_model1, av_model2, av_model3))
        # return (model1, model2, model3), model
        model = np.concatenate((av_model1, av_model2, av_model3, av_model4, av_model5))
        return (model1, model2, model3, model4, model5), model


class log_sub_cluster_ml_rscale_prob(object):
    def __init__(self, data, params, nbins=5, logmass_edges=(10, 18), c_edges=(2, 20), c_sub_edges=(5, 20), tau_pars=(0.17, 0.04), f_pars=(0.75, 0.08)):
        """logmass, c linear edges, tau, f gaussian prior (and zero cut)"""
        self.data = data
        self.params = params.params
        self.logmass_edges = logmass_edges
        self.c_edges = c_edges
        self.c_sub_edges = c_sub_edges
        self.tau_pars = tau_pars
        self.f_pars = f_pars
        self.BADVAL = BADVAL
        self.nbins = nbins

    def __call__(self, theta):
        """logmass, c, tau, f_cen"""
        logmass1, c1, ml, ralpha, tau_misc, f_cen = theta
        lp = 0
        cond = (((self.logmass_edges[0] > logmass1) or (self.logmass_edges[1] < logmass1)) or
                ((self.c_edges[0] > c1) or (self.c_edges[1] < c1)) or
                ((ml < 1e-3) or (ml > 1e4)) or
                ((f_cen > 1) or f_cen < 0)
                # ((self.logmass_edges[0] > logmass_bcg) or (self.logmass_edges[1] < logmass_bcg)) or
                # (10**logmass_bcg > 0.1*10**logmass1) or
                )
        if cond:
            # print("badval")
            lp = self.BADVAL
        else:
            y = self.data["dst"]
            cov = self.data["cov"]
            # print(y.shape)

            model = self.calc_model(theta)[1]
            # print(model.shape)
            dvec = y - model
            lp += -0.5 * np.dot(np.dot(dvec.T, np.linalg.inv(cov)), dvec)
            # print(lp)
            lp += -np.log(self.tau_pars[1]) - (tau_misc - self.tau_pars[0])**2 / (2 * self.tau_pars[1]**2)
            lp += -np.log(self.f_pars[1]) - (f_cen - self.f_pars[0])**2 / (2 * self.f_pars[1]**2)
            # print(lp)
        if not np.isfinite(lp):
            lp = self.BADVAL
        # if np.min(theta) < 0:
        #     lp = self.BADVAL
        return lp

    def calc_masses(self, ml, ralpha, rbin=0):
        lums = self.data["lsuns"][rbin]
        rpivot = self.data["rpivot"]
        dists = self.data["dists"][rbin]
        logmass = np.log10(ml * lums * (dists / rpivot)**ralpha).mean()
        # logmass = np.log10(ml * lums).mean()
        return logmass

    def calc_model(self, theta):
        logmass1, c1, ml, ralpha, tau_misc, f_cen = theta
        R_lambda = self.data["R_lambda"]
        R_misc = tau_misc * R_lambda

        bb = mass.calc_bias(logmass1, self.params)
        model1 = mass.calc_cluster_mix_nw(self.data["rarr_dense"][0], logmass1, c1, bb, R_misc, f_cen, self.params)
        av_model1 = averaging.average_profile_in_bins(self.data["rbin_edges"][0], self.data["rarr_dense"][0], model1)

        logmass2 = self.calc_masses(ml, ralpha, 0)
        c2 = mass.get_conc(logmass2, self.params)
        model2 = mass.calc_sub_mixture_nw(self.data["rarr_dense"][1], logmass1, c1, logmass2, c2, R_misc, f_cen, self.data["dists"][0], self.params, nbins=self.nbins)
        av_model2 = averaging.average_profile_in_bins(self.data["rbin_edges"][1], self.data["rarr_dense"][1], model2)

        logmass3 = self.calc_masses(ml, ralpha, 1)
        c3 = mass.get_conc(logmass3, self.params)
        model3 = mass.calc_sub_mixture_nw(self.data["rarr_dense"][2], logmass1, c1, logmass3, c3, R_misc, f_cen, self.data["dists"][1], self.params, nbins=self.nbins)
        av_model3 = averaging.average_profile_in_bins(self.data["rbin_edges"][2], self.data["rarr_dense"][2], model3)

        logmass4 = self.calc_masses(ml, ralpha, 2)
        c4 = mass.get_conc(logmass4, self.params)
        model4 = mass.calc_sub_mixture_nw(self.data["rarr_dense"][3], logmass1, c1, logmass4, c4, R_misc, f_cen, self.data["dists"][2], self.params, nbins=self.nbins)
        av_model4 = averaging.average_profile_in_bins(self.data["rbin_edges"][3], self.data["rarr_dense"][3], model4)

        logmass5 = self.calc_masses(ml, ralpha,3)
        c5 = mass.get_conc(logmass5, self.params)
        model5 = mass.calc_sub_mixture_nw(self.data["rarr_dense"][4], logmass1, c1, logmass5, c5, R_misc, f_cen, self.data["dists"][3], self.params, nbins=self.nbins)
        av_model5 = averaging.average_profile_in_bins(self.data["rbin_edges"][4], self.data["rarr_dense"][4], model5)
        # model = np.concatenate((av_model1, av_model2, av_model3))
        # return (model1, model2, model3), model
        model = np.concatenate((av_model1, av_model2, av_model3, av_model4, av_model5))
        return (model1, model2, model3, model4, model5), model



class log_sub_cluster_prob(object):
    def __init__(self, data, params, nbins=5, logmass_edges=(10, 18), c_edges=(2, 20), c_sub_edges=(5, 20), tau_pars=(0.17, 0.04), f_pars=(0.75, 0.08)):
        """logmass, c linear edges, tau, f gaussian prior (and zero cut)"""
        self.data = data
        self.params = params.params
        self.logmass_edges = logmass_edges
        self.c_edges = c_edges
        self.c_sub_edges = c_sub_edges
        self.tau_pars = tau_pars
        self.f_pars = f_pars
        self.BADVAL = BADVAL
        self.nbins = nbins

    def __call__(self, theta):
        """logmass, c, tau, f_cen"""
        logmass1, c1, logmass2, tau_misc, f_cen = theta
        lp = 0
        cond = (((self.logmass_edges[0] > logmass1) or (self.logmass_edges[1] < logmass1)) or
                ((self.logmass_edges[0] > logmass2) or (self.logmass_edges[1] < logmass2)) or
                ((self.c_edges[0] > c1) or (self.c_edges[1] < c1)) or
                ((f_cen > 1) or f_cen < 0)
                # ((self.logmass_edges[0] > logmass_bcg) or (self.logmass_edges[1] < logmass_bcg)) or
                # (10**logmass_bcg > 0.1*10**logmass1) or
                )
        if cond:
            # print("badval")
            lp = self.BADVAL
        else:
            y = self.data["dst"]
            cov = self.data["cov"]

            model = self.calc_model(theta)[1]
            dvec = y - model
            lp += -0.5 * np.dot(np.dot(dvec.T, np.linalg.inv(cov)), dvec)
            # print(lp)
            lp += -np.log(self.tau_pars[1]) - (tau_misc - self.tau_pars[0])**2 / (2 * self.tau_pars[1]**2)
            lp += -np.log(self.f_pars[1]) - (f_cen - self.f_pars[0])**2 / (2 * self.f_pars[1]**2)
            # print(lp)
        if not np.isfinite(lp):
            lp = self.BADVAL
        if np.min(theta) < 0:
            lp = self.BADVAL
        return lp

    def calc_model(self, theta):
        logmass1, c1, logmass2, tau_misc, f_cen = theta
        R_lambda = self.data["R_lambda"]
        R_misc = tau_misc * R_lambda

        bb = mass.calc_bias(logmass1, self.params)
        model1 = mass.calc_cluster_mix_nw(self.data["rarr_dense"][0], logmass1, c1, bb, R_misc, f_cen, self.params)
        av_model1 = averaging.average_profile_in_bins(self.data["rbin_edges"][0], self.data["rarr_dense"][0], model1)

        c2 = mass.get_conc(logmass2, self.params)
        model2 = mass.calc_sub_mixture_nw(self.data["rarr_dense"][1], logmass1, c1, logmass2, c2, R_misc, f_cen, self.data["dists"], self.params, nbins=self.nbins)
        av_model2 = averaging.average_profile_in_bins(self.data["rbin_edges"][1], self.data["rarr_dense"][0], model2)

        model = np.concatenate((av_model1, av_model2))
        return (model1, model2), model


class log_sub_cluster_prob2(object):
    def __init__(self, data, params, nbins=5,
                 logmass_edges=(10, 18), c_edges=(2, 20), c_sub_edges=(5, 20),
                 bcg_pars=(11.5, 1),
                 tau_pars=(0.17, 0.04), f_pars=(0.75, 0.08)):
        """logmass, c linear edges, tau, f gaussian prior (and zero cut)"""
        self.data = data
        self.params = params.params
        self.logmass_edges = logmass_edges
        self.c_edges = c_edges
        self.c_sub_edges = c_sub_edges
        self.bcg_pars = bcg_pars
        self.tau_pars = tau_pars
        self.f_pars = f_pars
        self.BADVAL = BADVAL
        self.nbins = nbins

    def __call__(self, theta):
        """logmass, c, tau, f_cen"""
        logmass1, c1, logmass2, logmass_bcg, tau_misc, f_cen = theta
        lp = 0
        cond = (((self.logmass_edges[0] > logmass1) or (self.logmass_edges[1] < logmass1)) or
                ((self.logmass_edges[0] > logmass2) or (self.logmass_edges[1] < logmass2)) or
                ((self.c_edges[0] > c1) or (self.c_edges[1] < c1)) or
                ((f_cen > 1) or f_cen < 0)
                )
        if cond:
            # print("badval")
            lp = self.BADVAL
        else:
            y = self.data["dst"]
            cov = self.data["cov"]

            model = self.calc_model(theta)[1]
            dvec = y - model
            lp += -0.5 * np.dot(np.dot(dvec.T, np.linalg.inv(cov)), dvec)
            # print(lp)
            lp += -np.log(self.tau_pars[1]) - (tau_misc - self.tau_pars[0])**2 / (2 * self.tau_pars[1]**2)
            lp += -np.log(self.f_pars[1]) - (f_cen - self.f_pars[0])**2 / (2 * self.f_pars[1]**2)
            lp += -np.log(self.bcg_pars[1]) - (logmass_bcg - self.bcg_pars[0])**2 / (2 * self.bcg_pars[1]**2)
        # print(lp)
        if not np.isfinite(lp):
            lp = self.BADVAL
        if np.min(theta) < 0:
            lp = self.BADVAL
        return lp

    def calc_model(self, theta):
        logmass1, c1, logmass2, logmass_bcg, tau_misc, f_cen = theta
        R_lambda = self.data["R_lambda"]
        R_misc = tau_misc * R_lambda

        bb = mass.calc_bias(logmass1, self.params)
        model1 = mass.calc_cluster_mix_nw(self.data["rarr_dense"][0], logmass1, c1, bb, R_misc, f_cen, self.params)
        av_model1 = averaging.average_profile_in_bins(self.data["rbin_edges"][0], self.data["rarr_dense"][0], model1)

        c2 = mass.get_conc(logmass2, self.params)
        c_bcg = mass.get_conc(logmass_bcg, self.params)
        model2 = mass.calc_sub_mixture2_nw(self.data["rarr_dense"][1], logmass1, c1, logmass2, c2,
                                           logmass_bcg, c_bcg, R_misc, f_cen, self.data["dists"],
                                           self.params, nbins=self.nbins)
        av_model2 = averaging.average_profile_in_bins(self.data["rbin_edges"][1], self.data["rarr_dense"][0], model2)

        model = np.concatenate((av_model1, av_model2))
        return (model1, model2), model


class log_sub_cluster_all_prob2(object):
    def __init__(self, data, params, nbins=5,
                 logmass_edges=(10, 18), c_edges=(2, 20), c_sub_edges=(5, 20),
                 bcg_pars=(11.5, 1),
                 tau_pars=(0.17, 0.04), f_pars=(0.75, 0.08)):
        """logmass, c linear edges, tau, f gaussian prior (and zero cut)"""
        self.data = data
        self.params = params.params
        self.logmass_edges = logmass_edges
        self.c_edges = c_edges
        self.c_sub_edges = c_sub_edges
        self.bcg_pars = bcg_pars
        self.tau_pars = tau_pars
        self.f_pars = f_pars
        self.BADVAL = BADVAL
        self.nbins = nbins

    def __call__(self, theta):
        """logmass, c, tau, f_cen"""
        logmass1, c1, logmass2, logmass3, logmass4, logmass5, logmass6, logmass_bcg, tau_misc, f_cen = theta
        lp = 0
        cond = (((self.logmass_edges[0] > logmass1) or (self.logmass_edges[1] < logmass1)) or
                ((self.c_edges[0] > c1) or (self.c_edges[1] < c1)) or
                ((f_cen > 1) or f_cen < 0)
                )
        if cond:
            # print("badval")
            lp = self.BADVAL
        else:
            y = self.data["dst"]
            cov = self.data["cov"]

            model = self.calc_model(theta)[1]
            dvec = y - model
            lp += -0.5 * np.dot(np.dot(dvec.T, np.linalg.inv(cov)), dvec)
            # print(lp)
            lp += -np.log(self.tau_pars[1]) - (tau_misc - self.tau_pars[0])**2 / (2 * self.tau_pars[1]**2)
            lp += -np.log(self.f_pars[1]) - (f_cen - self.f_pars[0])**2 / (2 * self.f_pars[1]**2)
            lp += -np.log(self.bcg_pars[1]) - (logmass_bcg - self.bcg_pars[0])**2 / (2 * self.bcg_pars[1]**2)
        # print(lp)
        if not np.isfinite(lp):
            lp = self.BADVAL
        if np.min(theta) < 0:
            lp = self.BADVAL
        return lp

    def calc_model(self, theta):
        logmass1, c1, logmass2, logmass3, logmass4, logmass5, logmass6, logmass_bcg, tau_misc, f_cen = theta
        R_lambda = self.data["R_lambda"]
        R_misc = tau_misc * R_lambda

        bb = mass.calc_bias(logmass1, self.params)
        model1 = mass.calc_cluster_mix_nw(self.data["rarr_dense"][0], logmass1, c1, bb, R_misc, f_cen, self.params)
        av_model1 = averaging.average_profile_in_bins(self.data["rbin_edges"][0], self.data["rarr_dense"][0], model1)

        c2 = mass.get_conc(logmass2, self.params)
        c3 = mass.get_conc(logmass3, self.params)
        c4 = mass.get_conc(logmass4, self.params)
        c5 = mass.get_conc(logmass5, self.params)
        c6 = mass.get_conc(logmass6, self.params)
        c_bcg = mass.get_conc(logmass_bcg, self.params)

        model2 = mass.calc_sub_mixture2_nw(self.data["rarr_dense"][1], logmass1, c1, logmass2, c2,
                                           logmass_bcg, c_bcg, R_misc, f_cen, self.data["dists"][0],
                                           self.params, nbins=self.nbins)
        av_model2 = averaging.average_profile_in_bins(self.data["rbin_edges"][1], self.data["rarr_dense"][0], model2)

        model3 = mass.calc_sub_mixture2_nw(self.data["rarr_dense"][1], logmass1, c1, logmass3, c3,
                                           logmass_bcg, c_bcg, R_misc, f_cen, self.data["dists"][1],
                                           self.params, nbins=self.nbins)
        av_model3 = averaging.average_profile_in_bins(self.data["rbin_edges"][1], self.data["rarr_dense"][0], model3)

        model4 = mass.calc_sub_mixture2_nw(self.data["rarr_dense"][1], logmass1, c1, logmass4, c4,
                                           logmass_bcg, c_bcg, R_misc, f_cen, self.data["dists"][2],
                                           self.params, nbins=self.nbins)
        av_model4 = averaging.average_profile_in_bins(self.data["rbin_edges"][1], self.data["rarr_dense"][0], model4)

        model5 = mass.calc_sub_mixture2_nw(self.data["rarr_dense"][1], logmass1, c1, logmass5, c5,
                                           logmass_bcg, c_bcg, R_misc, f_cen, self.data["dists"][3],
                                           self.params, nbins=self.nbins)
        av_model5 = averaging.average_profile_in_bins(self.data["rbin_edges"][1], self.data["rarr_dense"][0], model5)

        model6 = mass.calc_sub_mixture2_nw(self.data["rarr_dense"][1], logmass1, c1, logmass6, c6,
                                           logmass_bcg, c_bcg, R_misc, f_cen, self.data["dists"][4],
                                           self.params, nbins=self.nbins)
        av_model6 = averaging.average_profile_in_bins(self.data["rbin_edges"][1], self.data["rarr_dense"][0], model6)

        model = np.concatenate((av_model1, av_model2, av_model3, av_model4, av_model5, av_model6))
        # print(model.shape)
        return (model1, model2, model3, model4, model5, model6), model
    #

class log_sub_cluster_all_prob3(object):
    def __init__(self, data, params, nbins=5,
                 logmass_edges=(10, 18), c_edges=(2, 20), c_sub_edges=(5, 20),
                 bcg_pars=(11.5, 1),
                 tau_pars=(0.17, 0.04), f_pars=(0.75, 0.08)):
        """logmass, c linear edges, tau, f gaussian prior (and zero cut)"""
        self.data = data
        self.params = params.params
        self.logmass_edges = logmass_edges
        self.c_edges = c_edges
        self.c_sub_edges = c_sub_edges
        self.bcg_pars = bcg_pars
        self.tau_pars = tau_pars
        self.f_pars = f_pars
        self.BADVAL = BADVAL
        self.nbins = nbins

    def __call__(self, theta):
        """logmass, c, tau, f_cen"""
        logmass1, c1, logmass2, logmass3, logmass4, logmass5, logmass_bcg, tau_misc, f_cen = theta
        lp = 0
        cond = (((self.logmass_edges[0] > logmass1) or (self.logmass_edges[1] < logmass1)) or
                ((self.c_edges[0] > c1) or (self.c_edges[1] < c1)) or
                ((f_cen > 1) or f_cen < 0)
                )
        if cond:
            # print("badval")
            lp = self.BADVAL
        else:
            y = self.data["dst"]
            cov = self.data["cov"]

            model = self.calc_model(theta)[1]
            dvec = y - model
            lp += -0.5 * np.dot(np.dot(dvec.T, np.linalg.inv(cov)), dvec)
            # print(lp)
            lp += -np.log(self.tau_pars[1]) - (tau_misc - self.tau_pars[0])**2 / (2 * self.tau_pars[1]**2)
            lp += -np.log(self.f_pars[1]) - (f_cen - self.f_pars[0])**2 / (2 * self.f_pars[1]**2)
            lp += -np.log(self.bcg_pars[1]) - (logmass_bcg - self.bcg_pars[0])**2 / (2 * self.bcg_pars[1]**2)
        # print(lp)
        if not np.isfinite(lp):
            lp = self.BADVAL
        if np.min(theta) < 0:
            lp = self.BADVAL
        return lp

    def calc_model(self, theta):
        logmass1, c1, logmass2, logmass3, logmass4, logmass5, logmass_bcg, tau_misc, f_cen = theta
        R_lambda = self.data["R_lambda"]
        R_misc = tau_misc * R_lambda

        bb = mass.calc_bias(logmass1, self.params)
        model1 = mass.calc_cluster_mix_nw(self.data["rarr_dense"][0], logmass1, c1, bb, R_misc, f_cen, self.params)
        av_model1 = averaging.average_profile_in_bins(self.data["rbin_edges"][0], self.data["rarr_dense"][0], model1)

        c2 = mass.get_conc(logmass2, self.params)
        c3 = mass.get_conc(logmass3, self.params)
        c4 = mass.get_conc(logmass4, self.params)
        c5 = mass.get_conc(logmass5, self.params)
        c_bcg = mass.get_conc(logmass_bcg, self.params)

        model2 = mass.calc_sub_mixture2_nw(self.data["rarr_dense"][1], logmass1, c1, logmass2, c2,
                                           logmass_bcg, c_bcg, R_misc, f_cen, self.data["dists"][0],
                                           self.params, nbins=self.nbins)
        av_model2 = averaging.average_profile_in_bins(self.data["rbin_edges"][1], self.data["rarr_dense"][0], model2)

        model3 = mass.calc_sub_mixture2_nw(self.data["rarr_dense"][1], logmass1, c1, logmass3, c3,
                                           logmass_bcg, c_bcg, R_misc, f_cen, self.data["dists"][1],
                                           self.params, nbins=self.nbins)
        av_model3 = averaging.average_profile_in_bins(self.data["rbin_edges"][1], self.data["rarr_dense"][0], model3)

        model4 = mass.calc_sub_mixture2_nw(self.data["rarr_dense"][1], logmass1, c1, logmass4, c4,
                                           logmass_bcg, c_bcg, R_misc, f_cen, self.data["dists"][2],
                                           self.params, nbins=self.nbins)
        av_model4 = averaging.average_profile_in_bins(self.data["rbin_edges"][1], self.data["rarr_dense"][0], model4)

        model5 = mass.calc_sub_mixture2_nw(self.data["rarr_dense"][1], logmass1, c1, logmass5, c5,
                                           logmass_bcg, c_bcg, R_misc, f_cen, self.data["dists"][3],
                                           self.params, nbins=self.nbins)
        av_model5 = averaging.average_profile_in_bins(self.data["rbin_edges"][1], self.data["rarr_dense"][0], model5)

        model = np.concatenate((av_model1, av_model2, av_model3, av_model4, av_model5))
        # print(model.shape)
        return (model1, model2, model3, model4, model5), model


class log_cluster_prob(object):
    def __init__(self, data, params, logmass_edges=(10, 18), c_edges=(2, 20), c_sub_edges=(5, 20), tau_pars=(0.17, 0.04), f_pars=(0.75, 0.08), use_rcens=False):
        """logmass, c linear edges, tau, f gaussian prior (and zero cut)"""
        self.data = data
        self.params = params.params
        self.logmass_edges = logmass_edges
        self.c_edges = c_edges
        self.c_sub_edges = c_sub_edges
        self.tau_pars = tau_pars
        self.f_pars = f_pars
        self.BADVAL = BADVAL
        self.use_rcens = use_rcens

    def __call__(self, theta):
        """logmass, c, tau, f_cen"""
        logmass1, c1, tau_misc, f_cen = theta
        lp = 0
        cond = (((self.logmass_edges[0] > logmass1) or (self.logmass_edges[1] < logmass1)) or
                ((self.c_edges[0] > c1) or (self.c_edges[1] < c1)) or
                ((f_cen > 1) or f_cen < 0)
                )
        if cond:
            lp = self.BADVAL
        else:
            y = self.data["dst"]
            cov = self.data["cov"]

            model = self.calc_model(theta)[1]
            dvec = y - model
            lp += -0.5 * np.dot(np.dot(dvec.T, np.linalg.inv(cov)), dvec)
            lp += -np.log(self.tau_pars[1]) - (tau_misc - self.tau_pars[0])**2 / (2 * self.tau_pars[1]**2)
            lp += -np.log(self.f_pars[1]) - (f_cen - self.f_pars[0])**2 / (2 * self.f_pars[1]**2)
        if not np.isfinite(lp):
            lp = self.BADVAL
        if np.min(theta) < 0:
            lp = self.BADVAL
        return lp

    def calc_model(self, theta):
        logmass1, c1, tau_misc, f_cen = theta

        R_lambda = self.data["R_lambda"]
        R_misc = tau_misc * R_lambda
        bb = mass.calc_bias(logmass1, self.params)

        if self.use_rcens:
            model1 = mass.calc_cluster_mix_nw(self.data["rarr"], logmass1, c1, bb, R_misc, f_cen, self.params)
            av_model1 = model1
        else:
            model1 = mass.calc_cluster_mix_nw(self.data["rarr_dense"], logmass1, c1, bb, R_misc, f_cen, self.params)
            av_model1 = averaging.average_profile_in_bins(self.data["rbin_edges"], self.data["rarr_dense"], model1)

        return model1, av_model1

class log_nfw_prob(object):
    def __init__(self, data, params, logmass_edges=(10, 18), c_edges=(2, 20), c_sub_edges=(5, 20), tau_pars=(0.17, 0.04), f_pars=(0.75, 0.08)):
        """logmass, c linear edges, tau, f gaussian prior (and zero cut)"""
        self.data = data
        self.params = params.params
        self.logmass_edges = logmass_edges
        self.c_edges = c_edges
        self.c_sub_edges = c_sub_edges
        self.tau_pars = tau_pars
        self.f_pars = f_pars
        self.BADVAL = BADVAL

    def __call__(self, theta):
        """logmass, c, tau, f_cen"""
        logmass1, c1, = theta
        lp = 0
        cond = (((self.logmass_edges[0] > logmass1) or (self.logmass_edges[1] < logmass1)) or
                ((self.c_edges[0] > c1) or (self.c_edges[1] < c1)))
        if cond:
            lp = self.BADVAL
        else:
            y = self.data["dst"]
            cov = self.data["cov"]

            model = self.calc_model(theta)[1]
            dvec = y - model
            lp += -0.5 * np.dot(np.dot(dvec.T, np.linalg.inv(cov)), dvec)
        if not np.isfinite(lp):
            lp = self.BADVAL
        if np.min(theta) < 0:
            lp = self.BADVAL
        return lp

    def calc_model(self, theta):
        logmass1, c1 = theta

        model1 = mass.calc_nfw_nw(self.data["rarr_dense"], logmass1, c1, self.params)
        av_model1 = averaging.average_profile_in_bins(self.data["rbin_edges"], self.data["rarr_dense"], model1)
        return model1, av_model1

