
======================
Config files explained
======================

The aim of these config files is to automate running the weak lensing measurements and post-processing.
The module :py:data:`xpipe.paths` automatically tries to read the configs upon import, but you can also specify them later.

In the *pipeline* mode *xpipe* keeps track of reading lens catalogs, splitting them up into parameter bins,
measuring the lensing signal and estimating covariances and contaminations automatically. However in order to do this,
first we need to specify the details of these tasks in the below config files:

* params.yml_ defines the measurement parameters

* inputs.yml_ defines the available data files, and short aliases for them

These files do not exist yet when you first clone the repository, however there is a
``default_params.yml`` and ``default_inputs.yml`` which you should use as a reference. These defaults are set up such
that when you create ``params.yml`` and ``inputs.yml`` they will be automatically looked for and read.


Config files are defined as *yaml* files, and are read as dictionaries, each entry consisting
of a key and a corresponding value (note that this includes nested dictonaries).

Note that in *yaml* one should use ``null`` instead of ``None``

Load order
-----------

1) *default_params.yml*
2) looks for *params.yml*
3) tries to read :code:`custom_params_file` from *params.yml*

from this point the load is recursive, e.g. param files are loaded as long as there is a valid
custom params file defined in the last loaded config. Each new config file only *updates* the settings,
such that keys which are not present in the later files are left at their latest value.

The parameters defined here are loaded into the dictionary :py:data:`xpipe.paths.params`


.. _params.yml:

params.yml
------------


**Key reference**


* :py:data:`custom_params_file: params.yml`

    If you want to use an other parameter file, then specify it here. It must be in the same directory

* :py:data:`custom_data_path: False`

    Absolute path to the ``data`` directory of the pipeline. If False: uses ``default project_path + /data``


* :py:data:`mode: full`

    The pipeline supports two modes: ``full`` and ``dev``.
    This is primarily used in setting up the input files for the measurement. e.g. you can define two
    binning schemes: one really complex for the full run, and a simple, quicker for dev

* :py:data:`tag: default`

    Prefix for all files (with NO trailing "_"). In addition this will be the name of the directory
    wher input and output files are written to.


* :py:data:`shear_style: reduced`

    Format of the source galaxy catalog. Available formats are :code:`reduced`, :code:`lensfit` and
    :code:`metacal`

* :py:data:`cat_to_use: default`

    Alias for the lens catalog to be used (in this case the ``default``).
    Aliases are defined in inputs.yml_

* :py:data:`shear_to_use: default`

    Alias for the source catalog to be used (in this case the ``default``).
    Aliases are defined in inputs.yml_

* :py:data:`param_bins_full`

    Parameter bins defined for :code:`mode: full`, e.g.::

        param_bins_full:
            q0_edges: [0.2, 0.35, 0.5, 0.65]
            q1_edges: [5., 10., 14., 20., 30., 45., 60., 999]

    :code:`q0` and :code:`q1` refer to the *zero-th* and *first* *quantities* (in this order) you want to split
    your lens catalog by. For defining what these relate to see :py:data:`lenskey` and :py:data:`randkey`.
    In the above example :code:`q0`  is redshift, and :code:`q1` is optical richness.

    In general you can define an arbitrary number of quantities keeping the notation
    that the binning edges for quantity *n* are written as :code:`q[n]_edges`.


* :py:data:`param_bins_dev`

    Parameter bins defined for :code:`mode: dev`, e.g.::

        param_bins_dev:
            q0_edges: [0.2, 0.35]
            q1_edges: [45, 60]


    :code:`q0` and :code:`q1` refer to the *zero-th* and *first* *quantities* (in this order) you want to split
    your lens catalog by. For defining what these relate to see :py:data:`lenskey` and :py:data:`randkey`.
    In the above example :code:`q0`  is redshift, and :code:`q1` is optical richness.

    In general you can define an arbitrary number of quantities keeping the notation
    that the binning edges for quantity *n* are written as :code:`q[n]_edges`.


* :py:data:`lenskey`

    Aliases for the columns of the lens data table (assuming fits-like record table)::

        lenskey:
          id: MEM_MATCH_ID
          ra: RA
          dec: DEC
          z: Z_LAMBDA
          q0: Z_LAMBDA
          q1: LAMBDA_CHISQ

    :code:`q0` and :code:`q1` refer to the *zero-th* and *first* *quantities* (in this order) you want to split
    your lens catalog by (see :py:data:`param_bins_*`). In general you can define an arbitrary number of quantities keeping the notation
    that the alias for quantity *n* are written as :code:`q[n]`.
    In the above example :code:`q0`  is redshift, and :code:`q1` is optical richness.


* :py:data:`randkey`

    Aliases for the columns of the random points data table (assuming fits-like record table)::

        randkey:
          q0: ZTRUE
          q1: AVG_LAMBDAOUT
          ra: RA
          dec: DEC
          z: ZTRUE
          w: WEIGHT

    :code:`q0` and :code:`q1` refer to the *zero-th* and *first* *quantities* (in this order) you want to split
    your random points catalog by. In general you can define an arbitrary number of quantities keeping the notation
    that the alias for quantity *n* are written as :code:`q[n]`.
    In the above example :code:`q0`  is redshift, and :code:`q1` is optical richness

    **Note** that for random points you have to specify the *same* quantities as for the lens catalog.

* :py:data:`nprocess: 2`

    Number of *maximum* processes or CPU-s to use at the same time (OpenMP-style parallelization).

* :py:data:`njk_max: 100`

    Maximum number of Jackknife regions to use in resampling. Actual number is
    :code:`max(n_lens, njk_max)`

* :py:data:`nrandoms`

    Number of random points to use::

        nrandoms:
          full: 50000
          dev: 1000

* :py:data:`seeds`

    Random seed for choosing the random points :code:`random_seed`, and for generating rotated
    shear catalogs :code:`shear_seed_master`::

        seeds:
          random_seed: 5
          shear_seed_master: 10


* :py:data:`cosmo_params`

    Cosmology parameters defined as::

        cosmo_params:
          H0: 70.
          Om0: 0.3

* :py:data:`radial_bins`

    Logarithmic (base 10) radial bins from rmin to rmax::

        radial_bins:
          nbin: 15
          rmin: 0.0323
          rmax: 30.0
          units: Mpc

    Available units: :code:`Mpc`, :code:`comoving_mpc` or :code:`arcmin`


* :py:data:`weight_style: "optimal"`

    Source weight style in the **xshear** lensing measurement.
    Use :code:`optimal` when estimating :math:`\Delta\Sigma` and :code:`uniform` when measuring
    :math:`\gamma`.

* :py:data:`pairlog`

    Specifies the amount of source-lens pairs to be saved, and for which radial range::

        pairlog:
         pairlog_rmin: 0
         pairlog_rmax: 0
         pairlog_nmax: 0

    Note that the pair limit is considered for **each** call of *xshear* separately.
    That is if you separate lenses into Jackknife regions then this is applicable for a single region.

* :py:data:`lens_prefix: y1clust`

    Prefix for lens-files

* :py:data:`rand_prefix: y1rand`

    Prefix for random points files

* :py:data:`subtr_prefix: y1subtr`

    Prefix for lens - random points files

* :py:data:`fields_to_use: ['spt', 's82']`

    List of names of observational fields to use (as defined below)

* :py:data:`fields`

    Definition of observational field boundaries::

        fields:
          spt:
            dec_top: -30.
            dec_bottom: -60.
            ra_left: 0.
            ra_right: 360.
          s82:
            dec_top: 10.
            dec_bottom: -10.
            ra_left: 300.
            ra_right: 10.
          d04:
            dec_top: 10.
            dec_bottom: -30.
            ra_left: 10.
            ra_right: 250.

    These can be approximate, the only requirement is that they divide the lens dataset into
    the appropriate chunks

* :py:data:`pzpars`

    Parameters for the boost factor extraction::

        pzpars:
          hist:
            nbin: 15
            zmin: 0.0
            zmax: 3.0
            tag: "zhist"
          full:
            tag: "zpdf"
          boost:
            rbmin: 3
            rbmax: 13
            refbin: 14


    There are two modes histogram :py:data:`hist` which relies on Monte-Carlo samples of redshifts
    and is less robust, and :py:data:`full` which uses the full P(z) of each source galaxy.

    * :code:`tag` defines the name appended to the corresponding files.

    * :code:`boost` defines the radial range for the boost estimation in radial bins

* :py:data:`pdf_paths: null`

    Regular expression matching the absolute paths of the ``BPZ`` output files containing the full redshift PDF.
    (e.g. ``/home/data/*.h5``).

    **NOTE** This is only required for estimating the **Boost factors**, and can be safely left ``null`` in a simple
    lensing run.


.. _inputs.yml:

inputs.yml
----------


This config file lists the available data products. Currently all products are listed under the :py:data:`local`
key, indicating that they are found on disk, (as opposed to downloaded from some network location).

The two major sub-headings are:

* :py:data:`shearcat`

    Lists the available *xshear*-style source catalog files located within::

        [custom_data_path]/shearcat/

    where :py:data:`[custom_data_path]` is the absolute path to the :code:`data` folder
    specified by the corresponding key in params.yml_

    Each input file has it's *key* as an alias for the file name, such that you can use the key you
    define here for a valid value of :py:data:`shear_to_use` for params.yml_, e.g.::

          shearcat:
            default: default.dat
            im3shape: im3shape_shear_catalog.dat
            metacal: metacal_shear_catalog.dat


    These input files should be written in *ASCII*


* :py:data:`lenscat`

    Lists the available lens catalog files located within::

        [custom_data_path]/lenscat/

    where :py:data:`[custom_data_path]` is the absolute path to the :code:`data` folder
    specified by the corresponding key in params.yml_

    Each dataset has it's *key* as an alias, which you can use to define the lens dataset for a valid
    value of :py:data:`cat_to_use` for params.yml_. In addition, each dataset is implicitely assumed to
    consist of a **lens** catalog, and a corresponding catalog of **random points**, such that for each
    key there are two *sub-keys*: :py:data:`lens` and :py:data:`rand`. Both of these files should be written
    in *fits* format::

          lenscat:
            y1clust:
                lens: des_y1_lens_catalog.fits
                rand: des_y1_rand_catalog.fits
            svclust:
                lens: des_sv_lens_catalog.fits
                rand: des_sv_rand_catalog.fits
            testclust:
                lens: test_catalog.fits
                rand: null

    In case there are no random points available for the dataset you are using, it is safe to leave the
    :py:data:`rand` field empty, but in this case make sure you also use the :code:`--norands` flag
    when exectuing the pipeline scripts.


    In case the input catalog is defined in multiple files (for example when the parameter  bins are not trivial to
    define), a list of filenames can be defined for :py:data:`lens` and :py:data:`rand`::

        lenscat:
            pre_binned_data:
                lens: [
                    [des_y1_lens_catalog_bin-0-0.fits, des_y1_lens_catalog_bin-0-1.fits],
                    [des_y1_lens_catalog_bin-1-0.fits, des_y1_lens_catalog_bin-1-1.fits],
                ]

    **Note:** The defined files will be assumed to correspond to separat parameter selections, and thus
    this mode **cannot** be used together with the definiton of parameter bins in params.yml_
