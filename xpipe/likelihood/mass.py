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
import numpy as np
from cluster_toolkit import deltasigma
from cluster_toolkit import xi
from cluster_toolkit import miscentering
from cluster_toolkit import bias
from cluster_toolkit import averaging
from cluster_toolkit import concentration as conc
from classy import Class

import astropy.cosmology as cosmology

default_cosmo = cosmology.FlatLambdaCDM(Om0=0.3, H0=70)


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
            "Omega_b": self.Omega_b, # 0.3
            "h": self.h, # 0.7
            "P_lin": self.P_lin,
            "P_nonlin": self.P_nonlin,
            "k": self.k,
            "scale_factor": self.scale_factor,
            "logmass_array": logmass_array,
            "bias_array": self.bias_array,
            "n_s": self.classparams["n_s"]

        }


#####################################
# SIMPLE COMPONENTS


# def calc_bcg(rarr, logmass):
#     mass = 10**logmass
#
#     DeltaSigma = mass / (np.pi * (rarr*1e6)**2)
#     return DeltaSigma
#
#
# def calc_offset_bcg(rarr, logmass, dist):
#     mass = 10**logmass
#
#     # DeltaSigma = mass / (np.pi * (rarr*1e6)**2)
#     R_perp = np.logspace(-3, 3, 1000) #Mpc/h comoving; distance on the sky
#
#     # sigma
#     # return DeltaSigma


def calc_nfw_nw(rarr, logmass, c, params):
    factor = params["h"]/ params["scale_factor"]**2

    _scinv = params["scinv"] / factor

    _rarr = rarr * params["h"] / params["scale_factor"]
    mass = 10**logmass * params["h"]

    R_perp = np.logspace(-3, 3, 1000) #Mpc/h comoving; distance on the sky
    Sigma = deltasigma.Sigma_nfw_at_R(R_perp, mass, c, params["Omega_m"])
    Sigma2 = deltasigma.Sigma_nfw_at_R(_rarr, mass, c, params["Omega_m"])

    DeltaSigma = deltasigma.DeltaSigma_at_R(_rarr, R_perp, Sigma, mass, c, params["Omega_m"])

    DeltaSigma = DeltaSigma / (1 - Sigma2 * _scinv)
    #     factor = (1 / h) / (1 / h * scale_factor)**2
    factor = params["h"]/ params["scale_factor"]**2
    return DeltaSigma * factor

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


def calc_single_misc_nfw_nw(rarr, logmass, c, R_misc, params):
    _scinv = params["scinv"] / params["h"] * (params["h"] / params["scale_factor"])**2
    _rarr = rarr * params["h"] / params["scale_factor"]
    _R_misc = R_misc * params["h"] / params["scale_factor"]
    mass = 10**logmass * params["h"]
    # print(mass)
    R_perp = np.logspace(-3, 3, 1000) #Mpc/h comoving; distance on the sky
    Sigma = deltasigma.Sigma_nfw_at_R(R_perp, mass, c, params["Omega_m"])
    # print(Sigma.mean())
    # Sigma_misc = miscentering.Sigma_mis_single_at_R(_rarr, R_perp, Sigma, mass, c, params["Omega_m"], _R_misc)
    # return Sigma_misc
    Sigma_misc = miscentering.Sigma_mis_single_at_R(R_perp, R_perp, Sigma, mass, c, params["Omega_m"], _R_misc)
    Sigma_misc2 = miscentering.Sigma_mis_single_at_R(_rarr, R_perp, Sigma, mass, c, params["Omega_m"], _R_misc)

# # return Sigma_misc
    # # print(Sigma_misc.mean())
    DeltaSigma = miscentering.DeltaSigma_mis_at_R(_rarr, R_perp, Sigma_misc)
    DeltaSigma = DeltaSigma / (1 - Sigma_misc2 * _scinv)

# # print(DeltaSigma.mean())
    # #
    # factor = params["h"] / params["scale_factor"]**2
    return DeltaSigma# * factor


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


def calc_sub_misc_nfw(rarr, logmass, c, R_misc, f_cen, dist, params):
    _rarr = rarr * params["h"] / params["scale_factor"]
    _R_misc = R_misc * params["h"] / params["scale_factor"]
    _dist = dist * params["h"] / params["scale_factor"]
    # print(dist, _dist)
    mass = 10**logmass * params["h"]

    R_perp = np.logspace(-3, 2.4, 1000) #Mpc/h comoving; distance on the sky

    Sigma = deltasigma.Sigma_nfw_at_R(R_perp, mass, c, params["Omega_m"])
    Sigma_misc = miscentering.Sigma_mis_at_R(R_perp, R_perp, Sigma, mass, c, params["Omega_m"], _R_misc)
    Sigma_clust = f_cen * Sigma + (1 - f_cen) * Sigma_misc

    Sigma_sub_clust = miscentering.Sigma_mis_single_at_R(R_perp, R_perp, Sigma_clust, mass, c, params["Omega_m"], _dist)
    # print(Sigma_sub_clust)
    DeltaSigma = miscentering.DeltaSigma_mis_at_R(_rarr, R_perp, Sigma_sub_clust)

    factor = params["h"]/ params["scale_factor"]**2
    return DeltaSigma * factor

def calc_tnfw(rarr, logmass, c, rt, params):

    _rarr = rarr * params["h"] / params["scale_factor"]
    _rt = rt * params["h"] / params["scale_factor"]
    mass = 10**logmass * params["h"]

    radii = np.logspace(-2, 1, 100) #Mpc/h comoving
    xi_tnfw = xi.xi_nfw_at_r(radii, mass, c, params["Omega_m"])
    xi_tnfw[radii > _rt] *= 1e-6
    # print(xi_tnfw)
    R_perp = np.logspace(-2, 0.2, 100) #Mpc/h comoving; distance on the sky
    Sigma = deltasigma.Sigma_at_R(R_perp, radii, xi_tnfw, mass, c, params["Omega_m"])
    # print(Sigma)
    Sigma = np.nan_to_num(Sigma, )
    # print(Sigma)
    DeltaSigma = deltasigma.DeltaSigma_at_R(_rarr, R_perp, Sigma, mass, c, params["Omega_m"])
    # print(DeltaSigma)
    #
    factor = params["h"]/ params["scale_factor"]**2
    return DeltaSigma * factor


def calc_model(rarr, logmass, c, bias, params, component="both"):

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


def calc_bias(logmass, params):
    mass = 10**logmass * params["h"]

    b = bias.bias_at_M(mass, params["k"], params["P_lin"], params["Omega_m"])
    return b

def calc_cluster_mixture_model(x, logmass, c, R_misc, f_cen, params):
    """Cluster Assuming Gamma miscentering kernel"""

    # b = calc_bias(logmass, params)
    b = params["bias_array"][np.argmin((logmass - params["logmass_array"])**2)]

    model_cen = calc_model(x, logmass, c, b, params)
    model_miscen = calc_misc_model(x, logmass, c, b, R_misc, params)

    model = f_cen * model_cen + (1 - f_cen) * model_miscen
    return model


def calc_average_offset_nfw(rarr, logmass1, c1, R_misc, f_cen, distvals, params, nbins=4):

    vals, edges = np.histogram(distvals, bins=nbins)
    cens = edges[:-1] + np.diff(edges) / 2
    # print(cens, vals)

    profiles = []
    for cen in cens:
        # print(cen)
        tmp = calc_sub_misc_nfw(rarr, logmass1, c1, R_misc, f_cen, cen, params)
        profiles.append(tmp)
    model = np.average(profiles, axis=0, weights=vals)
    return model


def get_conc(logmass, params):
    mass = 10**logmass * params["h"]
    c = conc.concentration_at_M(mass, params["k"], params["P_nonlin"], params["n_s"],
                                params["Omega_b"], params["Omega_m"], params["h"])
    return c




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


def calc_sub_cluster_model(rbin_edges, rarr_dense, logmass1, c1, logmass2, logmass3, logmass4, c2, R_misc, f_cen, dists, params, dist_mult=1.42857):
    dist1, dist2, dist3 = dists
    rbc, rbs1, rbs2, rbs3 = rbin_edges
    xc, xs1, xs2, xs3 = rarr_dense
    # print(logmass1, c1, logmass2, logmass3, logmass4, c2, R_misc, f_cen, dist1, dist2, dist3)
    cluster_model = calc_cluster_mixture_model(xc, logmass1, c1, R_misc, f_cen, params)
    av_cluster_model = averaging.average_profile_in_bins(rbc, xc, cluster_model)
    # print(av_cluster_model.shape)
    sub_model1 = calc_nfw(xs1, logmass2, c2, params)
    sub_model2 = calc_nfw(xs2, logmass3, c2, params)
    sub_model3 = calc_nfw(xs3, logmass4, c2, params)
    # print(dists)
    offset_clust_model1 = calc_average_offset_nfw(xs1, logmass1, c1, R_misc, f_cen, dist1*dist_mult, params)
    offset_clust_model2 = calc_average_offset_nfw(xs2, logmass1, c1, R_misc, f_cen, dist2*dist_mult, params)
    offset_clust_model3 = calc_average_offset_nfw(xs3, logmass1, c1, R_misc, f_cen, dist3*dist_mult, params)
    # print(offset_clust_model3)
    # print(offset_clust_model2)
    # print(offset_clust_model1)
    #
    model1 = sub_model1 + offset_clust_model1
    av_model1 = averaging.average_profile_in_bins(rbs1, xs1, model1)
    model2 = sub_model2 + offset_clust_model2
    av_model2 = averaging.average_profile_in_bins(rbs2, xs2, model2)
    model3 = sub_model3 + offset_clust_model3
    av_model3 = averaging.average_profile_in_bins(rbs3, xs3, model3)
    # print(av_model1.shape, av_model2.shape, av_model3.shape)
    # print(av_model3)
    # # print(model1)
    model = np.concatenate((cluster_model, model1, model2, model3))
    average_model = np.concatenate((av_cluster_model, av_model1, av_model2, av_model3))
    # print(average_model.shape)
    # model = cluster_model
    return model, average_model



def calc_single_sub_clust_model(rbin_edges, rarr_dense, logmass1, logmass2, logmass_bcg, R_misc, f_cen, dist1, params):
    """Concentration set by Duffy"""
    rbc, rbs1 = rbin_edges
    xc, xs1  = rarr_dense

    c1 = get_conc(logmass1, params)
    c2 = get_conc(logmass2, params)
    cbcg = get_conc(logmass_bcg, params)
    cluster_model = calc_cluster_mixture_model(xc, logmass1, c1, R_misc, f_cen, params)
    bcg_model = calc_nfw(xc, logmass_bcg, cbcg, params)
    cluster_model = cluster_model + bcg_model
    av_cluster_model = averaging.average_profile_in_bins(rbc, xc, cluster_model)

    sub_model1 = calc_nfw(xs1, logmass2, c2, params)
    offset_clust_model1 = calc_average_offset_nfw(xs1, logmass1, c1, R_misc, f_cen, dist1, params)
    offset_bcg_model1 = calc_average_offset_nfw(xs1, logmass_bcg, cbcg, 0.1, 1, dist1, params)
    # offset_clust_model1 = calc_average_offset_nfw(xs1, logmass1, c1, R_misc, f_cen, dist1, params)

    model1 = sub_model1 + offset_clust_model1 + offset_bcg_model1
    av_model1 = averaging.average_profile_in_bins(rbs1, xs1, model1)

    model = np.concatenate((cluster_model, model1))
    average_model = np.concatenate((av_cluster_model, av_model1))
    return model, average_model


def calc_single_sub_clust_model2(rbin_edges, rarr_dense, logmass1, logmass2, logmass_bcg, R_misc, f_cen, dist1, params, nbin=5):
    """Concentration set by Duffy"""
    rbc, rbs1 = rbin_edges
    # print(rbc.shape, rbs1.shape)
    xc, xs1  = rarr_dense

    c1 = get_conc(logmass1, params)
    c2 = get_conc(logmass2, params)
    cbcg = get_conc(logmass_bcg, params)

    cluster_model = calc_cluster_mixture_model(xc, logmass1, c1, R_misc, f_cen, params)
    bcg_model = calc_nfw(xc, logmass_bcg, cbcg, params)
    cluster_model = cluster_model + bcg_model
    av_cluster_model = averaging.average_profile_in_bins(rbc, xc, cluster_model)

    sub_model1 = calc_nfw(xs1, logmass2, c2, params)
    offset_bcg_model1 = calc_average_offset_nfw(xs1, logmass_bcg, cbcg, R_misc, f_cen, dist1, params, nbins=nbin)
    offset_clust_model1 = calc_average_offset_nfw(xs1, logmass1, c1, R_misc, f_cen, dist1, params, nbins=nbin)

    model1 = sub_model1 + offset_clust_model1 + offset_bcg_model1
    av_model1 = averaging.average_profile_in_bins(rbs1, xs1, model1)
    # print(av_model1.shape)
    # print(av_cluster_model.shape)

    model = np.concatenate((cluster_model, model1))
    average_model = np.concatenate((av_cluster_model, av_model1))
    # print(average_model.shape)
    return model, average_model


###########################################################################################

def calc_cluster_mix_nw(rarr, logmass, c, bb, R_misc, f_cen, params, nonweak=True):
    """Galaxy cluster mixture model with nonweak shear correction"""
    factor = params["h"]/ params["scale_factor"]**2

    _scinv = params["scinv"] / factor

    _rarr = rarr * params["h"] / params["scale_factor"]
    _R_misc = R_misc * params["h"] / params["scale_factor"]

    mass = 10**logmass * params["h"]

    radii = np.logspace(-3, 3, 100) #Mpc/h comoving
    xi_nfw = xi.xi_nfw_at_r(radii, mass, c, params["Omega_m"])
    xi_mm = xi.xi_mm_at_r(radii, params["k"], params["P_nonlin"])
    xi_2halo = xi.xi_2halo(bb, xi_mm)

    xi_hm = xi.xi_hm(xi_nfw, xi_2halo)
    R_perp = np.logspace(-3, 2.4, 100) #Mpc/h comoving; distance on the sky
    Sigma_cen = deltasigma.Sigma_at_R(R_perp, radii, xi_hm, mass, c, params["Omega_m"])
    Sigma_misc = miscentering.Sigma_mis_at_R(R_perp, R_perp, Sigma_cen, mass, c, params["Omega_m"], _R_misc)

    Sigma = f_cen * Sigma_cen + (1 - f_cen) * Sigma_misc

    DeltaSigma = miscentering.DeltaSigma_mis_at_R(_rarr, R_perp, Sigma)

    if nonweak:
        _Sigma = np.interp(_rarr, R_perp, Sigma)
        DeltaSigma = DeltaSigma / (1 - _Sigma * _scinv)

    factor = params["h"]/ params["scale_factor"]**2
    return DeltaSigma * factor


def calc_sub_mixture_nw(rarr, logmass1, c1, logmass2, c2, R_misc, f_cen, distvals, params, nonweak=True, nbins=5):
    """Sat + Galaxy cluster mixture model with nonweak shear correction"""

    factor = params["h"]/ params["scale_factor"]**2
    _scinv = params["scinv"] / factor
    _rarr = rarr * params["h"] / params["scale_factor"]
    _R_misc = R_misc * params["h"] / params["scale_factor"]
    mass1 = 10**logmass1 * params["h"]
    mass2 = 10**logmass2 * params["h"]

    R_perp = np.logspace(np.log10(_rarr.min()*0.5), np.log10(_rarr.max()*1.5), 100) #Mpc/h comoving; distance on the sky
    Sigma1_cen = deltasigma.Sigma_nfw_at_R(R_perp, mass1, c1, params["Omega_m"])
    Sigma1_misc = miscentering.Sigma_mis_at_R(R_perp, R_perp, Sigma1_cen, mass1, c1, params["Omega_m"], _R_misc)
    Sigma1 = f_cen * Sigma1_cen + (1 - f_cen) * Sigma1_misc
    vals, edges = np.histogram(distvals, bins=nbins)
    cens = edges[:-1] + np.diff(edges) / 2

    off_Sigma1 = []
    for cen in cens:
        tmp = miscentering.Sigma_mis_single_at_R(R_perp, R_perp, Sigma1, mass1, c1, params["Omega_m"], cen)
        off_Sigma1.append(tmp)
    off_Sigma1 = np.average(off_Sigma1, axis=0, weights=vals)

    Sigma2 = deltasigma.Sigma_nfw_at_R(R_perp, mass2, c2, params["Omega_m"])

    Sigma = Sigma2 + off_Sigma1
    # return Sigma
    # print(R_perp.min(), _rarr.min())
    # print(R_perp.max(), _rarr.max())
    DeltaSigma = np.nan_to_num(miscentering.DeltaSigma_mis_at_R(_rarr, R_perp, Sigma))
    if nonweak:
        _Sigma = np.nan_to_num(np.interp(_rarr, R_perp, Sigma))
        DeltaSigma = DeltaSigma / (1 - _Sigma * _scinv)

    factor = params["h"]/ params["scale_factor"]**2
    return DeltaSigma * factor



def calc_sub_mixture2_nw(rarr, logmass1, c1, logmass2, c2, logmass_bcg, c_bcg, R_misc, f_cen, distvals, params, nonweak=True, nbins=5):
    """Sat + + BCG +  Galaxy cluster mixture model with nonweak shear correction"""

    factor = params["h"]/ params["scale_factor"]**2
    _scinv = params["scinv"] / factor
    _rarr = rarr * params["h"] / params["scale_factor"]
    _R_misc = R_misc * params["h"] / params["scale_factor"]
    mass1 = 10**logmass1 * params["h"]
    mass2 = 10**logmass2 * params["h"]
    mass_bcg = 10**logmass_bcg * params["h"]

    R_perp = np.logspace(-2.2, 2, 100) #Mpc/h comoving; distance on the sky
    Sigma1_cen = deltasigma.Sigma_nfw_at_R(R_perp, mass1, c1, params["Omega_m"])
    Sigma1_bcg = deltasigma.Sigma_nfw_at_R(R_perp, mass_bcg, c_bcg, params["Omega_m"])
    Sigma1_misc = miscentering.Sigma_mis_at_R(R_perp, R_perp, Sigma1_cen, mass1, c1, params["Omega_m"], _R_misc)
    Sigma1 = f_cen * Sigma1_cen + (1 - f_cen) * Sigma1_misc

    vals, edges = np.histogram(distvals, bins=nbins)
    cens = edges[:-1] + np.diff(edges) / 2
    off_Sigma1 = []
    for cen in cens:
        tmp1 = miscentering.Sigma_mis_single_at_R(R_perp, R_perp, Sigma1, mass1, c1, params["Omega_m"], cen)
        tmp2 = miscentering.Sigma_mis_single_at_R(R_perp, R_perp, Sigma1_bcg, mass_bcg, c_bcg, params["Omega_m"], cen)
        tmp = tmp1 + tmp2
        off_Sigma1.append(tmp)
    off_Sigma1 = np.average(off_Sigma1, axis=0, weights=vals)

    Sigma2 = deltasigma.Sigma_nfw_at_R(R_perp, mass2, c2, params["Omega_m"])

    Sigma = Sigma2 + off_Sigma1
    # return Sigma
    # print(Sigma)
    DeltaSigma = miscentering.DeltaSigma_mis_at_R(_rarr, R_perp, Sigma)
    if nonweak:
        _Sigma = np.interp(_rarr, R_perp, Sigma)
        DeltaSigma = DeltaSigma / (1 - _Sigma * _scinv)

    factor = params["h"]/ params["scale_factor"]**2
    return DeltaSigma * factor

def calc_sub_mixture2_clustonly_nw(rarr, logmass1, c1, logmass2, c2, logmass_bcg, c_bcg, R_misc, f_cen, distvals, params, nonweak=True, nbins=5):
    """Sat + + BCG +  Galaxy cluster mixture model with nonweak shear correction"""

    factor = params["h"]/ params["scale_factor"]**2
    _scinv = params["scinv"] / factor
    _rarr = rarr * params["h"] / params["scale_factor"]
    _R_misc = R_misc * params["h"] / params["scale_factor"]
    mass1 = 10**logmass1 * params["h"]
    mass2 = 10**logmass2 * params["h"]
    mass_bcg = 10**logmass_bcg * params["h"]

    R_perp = np.logspace(-2.2, 3, 100) #Mpc/h comoving; distance on the sky
    Sigma1_cen = deltasigma.Sigma_nfw_at_R(R_perp, mass1, c1, params["Omega_m"])
    Sigma1_bcg = deltasigma.Sigma_nfw_at_R(R_perp, mass_bcg, c_bcg, params["Omega_m"])
    Sigma1_misc = miscentering.Sigma_mis_at_R(R_perp, R_perp, Sigma1_cen, mass1, c1, params["Omega_m"], _R_misc)
    Sigma1 = f_cen * Sigma1_cen + (1 - f_cen) * Sigma1_misc

    vals, edges = np.histogram(distvals, bins=nbins)
    cens = edges[:-1] + np.diff(edges) / 2
    off_Sigma1 = []
    for cen in cens:
        tmp1 = miscentering.Sigma_mis_single_at_R(R_perp, R_perp, Sigma1, mass1, c1, params["Omega_m"], cen)
        tmp2 = miscentering.Sigma_mis_single_at_R(R_perp, R_perp, Sigma1_bcg, mass_bcg, c_bcg, params["Omega_m"], cen)
        tmp = tmp1 + tmp2
        off_Sigma1.append(tmp)
    off_Sigma1 = np.average(off_Sigma1, axis=0, weights=vals)

    # Sigma2 = deltasigma.Sigma_nfw_at_R(R_perp, mass2, c2, params["Omega_m"])

    Sigma = off_Sigma1
    # return Sigma
    # print(Sigma)
    DeltaSigma = miscentering.DeltaSigma_mis_at_R(_rarr, R_perp, Sigma)
    if nonweak:
        _Sigma = np.interp(_rarr, R_perp, Sigma)
        DeltaSigma = DeltaSigma / (1 - _Sigma * _scinv)

    factor = params["h"]/ params["scale_factor"]**2
    return DeltaSigma * factor


