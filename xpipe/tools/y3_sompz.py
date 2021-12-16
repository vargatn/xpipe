import astropy.constants as constants
import astropy.units as u
import astropy.cosmology as cosmology

import numpy as np
import h5py
import pandas as pd

default_cosmo = cosmology.FlatLambdaCDM(Om0=0.3, H0=70)


def sigma_crit_inv(zclust, z, cosmo=default_cosmo):
    """
    Calculates sigma crit inverse

    Parameters
    ----------
    zclust : float
        cluster redshift
    z : float
        source redshift
    cosmo : astropy.cosmology.FlatLambdaCDM
        cosmology container

    Returns
    -------
    resval : float
        Sigma_Crit inverse in pc^2 / Msun

    """
    prefac = (4. * np.pi * constants.G / (constants.c**2.)).to(u.pc / u.Msun)

    Ds = cosmo.angular_diameter_distance(z).to(u.pc)
    Dl = cosmo.angular_diameter_distance(zclust).to(u.pc)
    Dls = cosmo.angular_diameter_distance_z1z2(zclust, z).to(u.pc)

    if Ds != 0.:
        val = prefac * Dl * Dls / Ds
    else:
        val = prefac * 0.

    resval = np.max((0. , val.value))
    return resval



class sompz_reader(object):
    """Reader class for DES Y3 - style SOMPZ data"""
    def __init__(self, main_file_path, z_clust_grid_edges=None):
        """
        Reader class for DES Y3 - style SOMPZ data, also loads BPZ Z_MC redshifts

        Parameters
        ----------
        main_file_path : str
            file path of the Y3 master catalog
        z_clust_grid_edges : np.array
            list of redshift histogram edges for cluster sample
        """

        self.main_file = h5py.File(main_file_path, mode = 'r')
        zlows = self.main_file["catalog/sompz/pzdata/zlow"][:]
        zhighs = self.main_file["catalog/sompz/pzdata/zhigh"][:]
        self.zcens = zlows + (zhighs - zlows) / 2.
        self.zcens = self.zcens[:300]
        # zedges = np.concatenate((zlows[:300],  [zhighs[299],]))

        self.source_bins = [
            self.main_file["catalog/sompz/pzdata/bin0"][:],
            self.main_file["catalog/sompz/pzdata/bin1"][:],
            self.main_file["catalog/sompz/pzdata/bin2"][:],
            self.main_file["catalog/sompz/pzdata/bin3"][:],
        ]

        if z_clust_grid_edges is not None:
            self.zclust_grid_edges = z_clust_grid_edges
        else:
            self.zclust_grid_edges = np.linspace(0.00, 1.0, 101)
        self.zclust_grid = self.zclust_grid_edges[:-1] + np.diff(self.zclust_grid_edges) / 2.

        self.select = self.main_file["index/select"][:]
        self.cid = self.main_file["catalog/bpz/unsheared/coadd_object_id"][:].byteswap().newbyteorder()
        self.zmc = self.main_file["catalog/bpz/unsheared/zmc_sof"][:].byteswap().newbyteorder()
        self.pz_chat = self.main_file["catalog/sompz/pzdata/pz_chat"][:]
        self.bpz = pd.DataFrame()
        self.bpz["ID"] = self.cid[self.select]
        self.bpz["ZMC"] = self.zmc[self.select]

    def build_lookup(self, verbose=False):
        """Pre-calculates the sigma_crit_inverse values for a source histogram mid-points tomographic redshift bins"""
        self.scritinv_tab = np.zeros(shape=(len(self.zclust_grid), len(self.zcens)))
        for i, zclust in enumerate(self.zclust_grid):
            if verbose & (i%10==0):
                # print(verbose)
                print(i)
            for j, zsource in enumerate(self.zcens):
                self.scritinv_tab[i,j] = sigma_crit_inv(zclust, zsource)

    def get_single_scinv(self, clust_zval, sbin=-1):
        """Calculates sigma_crit_inverse with a single lens redshift and a tomographic source bin"""
        ii = np.argmin((clust_zval -  self.zclust_grid)**2.)

        scvals = np.average(self.scritinv_tab[ii], weights=self.source_bins[sbin])

        return scvals

    def get_bin_scinv(self, clust_zvals, sbin=-1, weights=None):
        """Calculates sigma_crit_inverse with a single lens redshift and a tomographic source bin"""
        try:
            len(clust_zvals)
        except:
            clust_zvals = (clust_zvals,)
        ccounts = np.histogram(clust_zvals, bins=self.zclust_grid_edges, weights=weights)[0]

        scvals = np.zeros(len(self.zclust_grid))
        for i, z in enumerate(self.zclust_grid):
            scvals[i] = np.average(self.scritinv_tab[i], weights=self.source_bins[sbin])
        scritinv = np.average(scvals, weights=ccounts)
        return scritinv






