import numpy as np

import xpipe.likelihood.mcmc
import xpipe.paths as paths
import xpipe.likelihood.mass as mass
import xpipe.likelihood.mcmc as mcmc


import pickle

import astropy.cosmology as cosmology
# this is just the default cosmology

# we make sure the correct config file is loaded here, it will let us automatically now what type of files
# were / will be produced, and where they will be placed
paths.update_params("/home/moon/vargatn/DES/PROJECTS/xpipe/settings/params_y3rm-sub_meta.yml")

# z = ACP.target["Z_LAMBDA"].mean()
z = 0.33
cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)
params = mass.make_params(z, cosmo)

for rbin in np.arange(5):
    data = pickle.load(open("/home/moon/vargatn/DES/PROJECTS/1_PROJECT_Subhalo_joint-dev/November_sprint/concat_sub_profiles_tmp_v2_rbin{}.p".format(rbin), "rb"))
    theta_init = (14.7, 12.2, 13., 0.17, 0.9)
    lprob = mcmc.log_sub_cluster_prob(data, params)

    lprob(theta_init)

    flat_samples, sampler = mcmc.do_mcmc(lprob, theta_init, nwalkers=32, nstep=500)

    print("success!!!")
    pickle.dump(flat_samples, open("/home/moon/vargatn/DES/PROJECTS/1_PROJECT_Subhalo_joint-dev/November_sprint/sub_flat_samples_tmp_03_rbin{}.p".format(rbin), "wb"))
    pickle.dump(sampler, open("/home/moon/vargatn/DES/PROJECTS/1_PROJECT_Subhalo_joint-dev/November_sprint/sub_sampler_tmp_03_rbin{}.p".format(rbin), "wb"))


    data = pickle.load(open("/home/moon/vargatn/DES/PROJECTS/1_PROJECT_Subhalo_joint-dev/November_sprint/concat_sub_profiles_tmp_v2_rbin{}_short.p".format(rbin), "rb"))
    theta_init = (14.7, 12.2, 13., 0.17, 0.9)
    lprob = mcmc.log_sub_cluster_prob(data, params)

    lprob(theta_init)

    flat_samples, sampler = mcmc.do_mcmc(lprob, theta_init, nwalkers=32, nstep=500)

    print("success!!!")
    pickle.dump(flat_samples, open("/home/moon/vargatn/DES/PROJECTS/1_PROJECT_Subhalo_joint-dev/November_sprint/sub_flat_samples_tmp_03_rbin{}_short.p".format(rbin), "wb"))
    pickle.dump(sampler, open("/home/moon/vargatn/DES/PROJECTS/1_PROJECT_Subhalo_joint-dev/November_sprint/sub_sampler_tmp_03_rbin{}_short.p".format(rbin), "wb"))
