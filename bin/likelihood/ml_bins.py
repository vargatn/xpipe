import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl

import h5py

import healpy as hp

import scipy
import copy

import xpipe.tools.catalogs as catalogs
import xpipe.paths as paths
import xpipe.xhandle.parbins as parbins
import xpipe.xhandle.xwrap as xwrap
import xpipe.tools.selector as selector
import xpipe.xhandle.shearops as shearops
import xpipe.xhandle.pzboost as pzboost
import xpipe.tools.y3_sompz as sompz
import xpipe.tools.visual as visual

import xpipe.likelihood.mass as mass
import xpipe.likelihood.mcmc as mcmc



from importlib import reload
import pickle

import multiprocessing as mp
from multiprocessing import Pool
import emcee

import cluster_toolkit.bias as bias

import astropy.cosmology as cosmology
# this is just the default cosmology
cosmo = cosmology.FlatLambdaCDM(Om0=0.3, H0=70)

# we make sure the correct config file is loaded here, it will let us automatically now what type of files
# were / will be produced, and where they will be placed
paths.update_params("/home/moon/vargatn/DES/PROJECTS/xpipe/settings/params_y3rm-sub-clust_meta.yml")

fname_root = "/home/moon/vargatn/DES/PROJECTS/1_PROJECT_Subhalo_joint-dev/December_sprint/"
# main_file_path = "/e/ocean1/users/vargatn/DESY3/Y3_mastercat_03_31_20.h5"
# src = sompz.sompz_reader(main_file_path)
# src.build_lookup()

for ibin in np.arange(4):
    print("ibin = ", ibin)
    data = pickle.load(open(fname_root +"concat_sub_profiles_v3_ibin{}_All.p".format(ibin), "rb"))
    profiles = data["sub_profiles"]
    cprof = data["clust_profile"]
    distvals = data["distvals"]
    targets = data["targets"]

    # z = data["targets"][0]["Z_LAMBDA"].mean()
    # scinvs = [
    #     src.get_single_scinv(z, sbin=2),
    #     src.get_single_scinv(z, sbin=3),
    # ]
    # scinv = np.mean(scinvs)
    cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)
    params = mass.make_params(z, cosmo)
    params.params.update({"scinv": scinv})

    cedges = np.logspace(np.log10(0.1), np.log10(100), 16)
    sedges = np.logspace(np.log10(0.04), np.log10(5), 11)
    rmins = [0.15, 0.35, 0.35, 0.35]

    rarr = np.logspace(np.log10(0.02), 2, 100)
    rarr_dense = [rarr] *2

    scens = sedges[:-1] + np.diff(sedges) / 2.
    tmp_data_subs = []
    irvals = []
    lsuns = []
    for i, rbin in enumerate((1, 2, 3, 4)):
        #         print(rmins[i])
        ir = np.where(scens < rmins[i])[0][-1] + 1
        #         print(scens[:ir])

        rbin_edges = [
            cedges,
            sedges[:ir+1],
        ]

        tmp = {
            "dst": np.concatenate((cprof.dst, profiles[rbin].dst[:ir])),
            "cov": scipy.linalg.block_diag(cprof.dst_cov, profiles[0].dst_cov[:ir, :][:, :ir]),
            "R_lambda": targets[rbin]["R_LAMBDA"].mean(),
            "rbin_edges": rbin_edges,
            "rarr_dense": rarr_dense,
            "dists": distvals[rbin],
            "lsuns": 10**(((targets[rbin]["MODEL_MAG_2_MEMB"] - cosmo.distmod(0.33)) - 4.57) / (-2.5)).values
        }
        tmp_data_subs.append(tmp)

    print("starting MCMC")
    # reload(mcmc)
    theta_init = (14.7, 4.5, 100, 0.2, 0.8)
    flat_samples = []
    samplers = []
    for i, rbin in enumerate((1, 2, 3, 4)):
        tmp = tmp_data_subs[i]
        lprob = mcmc.log_sub_cluster_ml_prob(tmp_data_subs[i], params)
        lprob(theta_init)
        flat_sample, sampler =  mcmc.do_mcmc(lprob, theta_init, nstep=2000, discard=1000)
        flat_samples.append(flat_sample)
        samplers.append(sampler)

    pickle.dump(flat_samples, open(fname_root + "dev-02_flat_samples_v04_ibin{}_All.p".format(ibin), "wb"))
    pickle.dump(samplers, open(fname_root + "dev-02_samplers_v04_ibin{}_All.p".format(ibin), "wb"))
    pickle.dump(tmp_data_subs, open(fname_root + "dev-02_data_subs_v04_ibin{}_All.p".format(ibin), "wb"))