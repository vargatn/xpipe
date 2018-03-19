
============================
Weak lensing pipeline guide
============================


This is a brief introduction on how to use this package in **pipeline** mode. Please note that
this Tutorial describes a simple scenario, in case you encounter problems or unexpected behaviour,
inspect the source code, or contact us directly.


Measuring the weak lensing data vector
--------------------------------------

Some pre-defined scripts are located in :py:data:`bin/redpipe/`


1.  Define the parameters and inputs as described in :doc:`Config files explained <config>`

|

2.  Exectute :py:data:`mkbins.py`, there are some flags available, e.g. in case you don't have any
    random points, you can use the :code:`--norands` flag to skip them.

    This script loads the input files, and splits them into parameter bins and JK-patches, and writes
    them to disk in a format which is understood by XSHEAR

    The input files are written to :py:data:`[custom_data_path]/xshear_in/[tag]/`

|

3.  Run XSHEAR on the created input files. Depending on the choice of source galaxy
    catalog use either :py:data:`xshear.py` for normal runs, and :py:data:`xshear_metacal.py`
    for METACALIBRATION.

    **Note** that this step might take a very long time, consider running it on a dedicated
    computing cluster

    This step support OpenMP style parralelization to assign the calculation of separate K-means
    regions to multiple cores. As a backup solution, it also supports splitting it up to multiple
    individual tasks via the flags :code:`--nchunk` (number of chunks), and :code:`--ichunk`
    (ID of chunks).

    The output files are written to :py:data:`[custom_data_path]/xshear_out/[tag]/`

|

4.  Extract the lensing profile from the xshear results via :py:data:`postprocess.py`

    By default the extracted quantity is :math:`\Delta\Sigma`, but :math:`\gamma_t` can also be
    extracted by re-defining attributes of :py:data:`StackedProfileContainer`.

    The results are written to :py:data:`[custom_data_path]/results/[tag]/`

    The resulting lensing profiles are written as :code:`_profile.dat`, and the corresponding
    Jackknife covariance is saved as :code:`_cov.dat`.

    In case random points are also defined, there are three types of output files: **lens**,
    **randoms** and **subtracted**.


Boost factor estimates from P(Z) decomposition
----------------------------------------------

* describe pair-logging

* describe pwsum-creation

* describe P(z) combining

* describe decomposition



