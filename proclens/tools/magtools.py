"""
Apparent to Absolute magnitude conversion **WORK IN PROGRESS**
"""

import numpy as np
import astropy.cosmology
import copy
import scipy.interpolate as interp
import scipy.integrate as integr
import pickle


class AbsMagConverter(object):
    def __init__(self, cosmo="default"):
        """
        Apparent to Absolute magnitude conversion for red galaxies

        Uses distance modulus + k-correction with single template


        Examples
        --------

        without saving lookup table::

            # response function of the survey
            resp = {
                "survey": "DES",  # name string for survey
                "lambda": lambda_array,  # array of wavelength values
                "lambda unit": "nm"  #  this must be the same for the response and
                                     #  the template
                "bands": bands #  dictionary of band name, response array pairs
                               #  bands = {"r": r_arr, "g": g_arr}
            }

            # template spectra for the conversion
            templ = {
                "type": "red quiescent",
                "lambda": lambda_array,  # array of wavelength values
                "lambda unit": "nm",  #  this must be the same for the response and
                                      #  the template
                "sed": sed,  # Values corresponding to the lambda values
                "sed unit": "Normalized Flux" # this is OK as long as the norm flag
                                              # is on when specifying the template
            }

            aconv = AbsMagConverter(cosmo=astropy.cosmology.FlatLambdaCDM(H0=70, Om0=0.3))
            aconv.add_response(resp)
            aconv.add_template(templ, norm=True)

            # here mag is the set of apparent magnitudes, and z is the redshift
            # of each object. band is the string name of the band, like "r"
            absmag = aconv.convert(self, mag, z, band, z2=0.0, num=100,
                                   z_0='auto', z_1='auto', use_tables=False)

            # after the conversion is complete we can save the used lookup table
            # for later use with a pickle!

            aconv.save_tables("mytable.p")

        Parameters
        ----------
        cosmo : astropy.cosmology.FlatLambdaCDM
            cosmology to use, default: H0=70, Om0=0.3
        """

        # getting cosmology
        if cosmo == "default":
            cosmo = astropy.cosmology.FlatLambdaCDM(H0=70, Om0=0.3)
        assert isinstance(cosmo, astropy.cosmology.FlatLambdaCDM)
        self.cosmo = cosmo

        # survey parameters
        self.survey = None
        self.rlamb = None
        self.rlamb_u = None
        self.bands = None

        # template parameters
        self.type = None
        self.tlamb = None
        self.tlamb_u = None
        self.sed = None
        self.sed_u = None

        # kcorrection parameters
        self.kind = None
        self.fresp = None
        self.ftemp = None
        self.x1 = None
        self.x2 = None
        self.ctab = None
        self.ifunc = None

        # reference table variables
        self.dtab = None

    @classmethod
    def from_table(cls, table):
        """Loads class instance using a pre-calculated table for each band"""
        cosmo = astropy.cosmology.FlatLambdaCDM(**table['cosmo pars'])
        amag = cls(cosmo)
        amag.add_response(table['response'])
        amag.add_template(table['template'])
        amag.dtab = table
        return amag

    @classmethod
    def from_template(cls, template, response, cosmo="default"):
        """Loads class instance using a particular template and response"""
        amag = cls(cosmo)
        amag.add_response(response)
        amag.add_template(template)
        return amag

    def add_response(self, rdict):
        """Loads response curves for different bands"""
        assert isinstance(rdict, dict)
        tmp = copy.copy(rdict)
        self.survey = tmp.pop('survey')
        self.rlamb = tmp.pop('lambda')
        self.rlamb_u = tmp.pop('lambda unit')
        self.bands = tmp

    def add_template(self, tdict, norm=True):
        """Loads SED template"""
        assert isinstance(tdict, dict)
        tmp = copy.copy(tdict)
        self.type = tmp['type']
        self.tlamb = tmp['lambda']
        self.tlamb_u = tmp['lambda unit']
        self.sed = tmp['sed']
        self.sed_u = tmp['sed unit']
        if norm:
            self.sed /= np.max(self.sed)
            self.sed_u = "normalized by max"

    def dist_modulus(self, z):
        """Calculates distance modulus for redshift z"""
        return self.cosmo.distmod(z).value

    def _bandselect(self, band, **kwargs):
        """creates interpolating function for band"""
        yy = self.bands[band]
        self.kind = band
        self.x1, self.x2 = self._getlim(self.rlamb, yy, **kwargs)
        self.fresp = self._interp(self.rlamb, yy, **kwargs)

    def kcorrect(self, z, band, z2=0.0):
        """
        Calculates K-correction from band --> band

        Parameters
        ----------
        z : float or np.array
            redshifts
        band : str
            One of the available bands
        z2 : float
            observer's redshift

        Returns
        -------
        float or np.array
            k-korrection
        """
        # creating template function
        self.ftemp = self._interp(self.tlamb, self.sed)
        # creating response function
        self._bandselect(band)

        if not np.iterable(z):
            kval = self._kint(z, z2)
        else:
            kval = np.array([self._kint(zval, z2) for zval in z])

        return kval

    def _kint(self, z1, z2=0.0, **kwargs):
        """Calculates k-correction for redshift z"""
        assert self.ftemp is not None
        assert self.fresp is not None
        err_msg = "template and response units incompatible!, check units..."
        assert self.tlamb_u == self.rlamb_u, err_msg
        oint, oerr = self._windowint(z1, self.x1, self.x2)
        eint, eerr = self._windowint(0.0, self.x1, self.x2, z2=z2)
        kval = -2.5 * np.log10(oint / eint / (1. + z1))
        return kval

    @staticmethod
    def _interp(xarr, yarr, **kwargs):
        """creates interpolator function using scipy linear interp"""
        return interp.interp1d(xarr, yarr, bounds_error=False, fill_value=0.0)

    @staticmethod
    def _getlim(xx, yy, ralim=1e-4):
        """
        Gets x edges where yy is larger than ralim-th of the full amplitude

        Parameters:
        -----------
        xx : np.array
            x-values
        yy : np.array
            y-values
        ralim : float
            relative amplitude limit

        Returns
        -------
        float, float
            x0, x1
        """
        yresc = yy / np.max(yy)
        inds = np.where(yresc > ralim)
        ind0 = np.min(inds)
        ind1 = np.max(inds)
        return xx[ind0], xx[ind1]

    @staticmethod
    def _frexpr(ll, ftempl, fresp, z=0.0, z2=0.0):
        """
        Argument in frame at redshift z

        Parameters
        ----------
        ll : float
            wavelength [nm]
        ftempl : float
            emitter SED function
        fresp : float
            detector response function
        z : float
            redshift

        Returns
        -------
        float
            value of integrand
        """
        val =  ll * ftempl(ll / (1. + z)) * fresp(ll * (1. + z2))
        return val

    def _windowint(self, z, low, high, z2=0.0, **kwargs):
        """Integral over the spectra in frame at z"""
        assert self.ftemp is not None
        assert self.fresp is not None
        y, abserr = integr.quad(self._frexpr, low *(1. + z2), high * (1. + z2),
                                args=(self.ftemp, self.fresp, z, z2))
        return y, abserr

    def _conv_table(self, zarr, band, z2):
        """
        Calculates a lookup table for quick calculations

        Parameters
        ----------
        zarr : np.array
            specified z values
        band : str
            name of the band
        z2 : float
            observer redshift
        """
        dmod = self.dist_modulus(zarr)
        kval = self.kcorrect(zarr, band, z2)
        self.ctab = np.vstack((zarr, dmod + kval, dmod, kval)).T

    def _get_ifunc(self):
        """Creates correction interpolator"""
        assert self.ctab is not None
        self.ifunc = interp.interp1d(self.ctab[:, 0], self.ctab[:, 1])

    def _querry(self, z):
        """Querries lookup table with z"""
        assert self.ctab is not None
        assert self.ifunc is not None
        return self.ifunc(z)

    @staticmethod
    def _calc_zarr(z, z_0='auto', z_1='auto', num=100):
        """Guesses an appropriate z-array"""
        assert np.iterable(z)
        if z_0 == 'auto':
            z_0 = np.min(z)
        if z_1 == 'auto':
            z_1 = np.max(z)
        return np.linspace(z_0, z_1, num)

    def convert(self, mag, z, band, z2=0.0, num=100, z_0='auto', z_1='auto',
                use_tables=True, **kwargs):
        """
        Converts apparent magnitudes to absolute

        Parameters
        ----------
        mag : np.array
            observed magnitudes
        z : np.array
            redsift of objects
        band : str
            name of band
        z2 : float
            redshift of observed
        use_tables : bool
            try to use pre calculated table for the band
        num : int
            number of points in reference tables (for auto z array)
        z_0 : float
            start of redshift table
        z_1 : float
            end of redshift table

        Returns
        -------
        np.array
            absolute magnitudes
        """

        if use_tables and np.abs(z2 - self.dtab['z2']) < 1e-3:
            assert self.dtab is not None
            self.ctab = self.dtab[band]
        else:
            # calculating lookup redshifts
            zarr = self._calc_zarr(z, z_0, z_1, num)
            # building conversion table
            self._conv_table(zarr, band, z2)

        # querryiong the conversion table
        self._get_ifunc()
        korr = self._querry(z)

        return mag - korr

    def build_tables(self, zarr, z2=0.0, **kwargs):
        """
        Creates a reference table for all the bands

        Loop throught all bands available and build a table for each
        then assemble this to a dict with other relevant info

        Parameters
        ----------
        zarr : np.array
            z values to use
        z2 : float
            observer redshift
        """

        # reconstructing response dictionary
        respdict = {
            'survey': self.survey,
            'lambda': self.rlamb,
            'lambda unit': self.rlamb_u,
        }
        respdict.update(self.bands)

        # reconstructing template dictionary
        tempdict = {
            'type': self.type,
            'lambda': self.tlamb,
            'lambda unit': self.tlamb_u,
            'sed': self.sed,
            'sed unit': self.sed_u
        }

        assert isinstance(self.cosmo, astropy.cosmology.FlatLambdaCDM)
        cosmo_pars = {
            'H0': self.cosmo.H0,
            'Om0': self.cosmo.Om0,
            'Tcmb0': self.cosmo.Tcmb0,
            'Neff': self.cosmo.Neff,
            'm_nu': self.cosmo.m_nu,
            'Ob0': self.cosmo.Ob0,
        }

        colnames = ['z', 'dmod+kcorr', 'dmod', 'kcorr']
        tabdict = {
            'response': respdict,
            'template': tempdict,
            'colnames': colnames,
            'cosmo pars': cosmo_pars,
            'z2': z2,
        }

        # building up the correction table for each band
        bands = list(self.bands.keys())
        for band in bands:
            print(band)
            self._conv_table(zarr, band, z2=z2)
            tabdict.update({band: copy.deepcopy(self.ctab)})

        self.dtab = tabdict

    def save_tables(self, tname):
        """ saves table dict to pickle"""
        pickle.dump(copy.deepcopy(self.dtab), open(tname, 'wb'))