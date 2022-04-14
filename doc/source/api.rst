

=============
API Reference
=============


DES Y3 wrappers
---------------

SOMPZ
"""""

.. autosummary::
    :toctree: generated

    xpipe.tools.y3_sompz.sigma_crit_inv

.. autosummary::
    :toctree: generated
    :template: custom.rst

    xpipe.tools.y3_sompz.sompz_reader

Combine shear profiles and apply calibrations
"""""""""""""""""""""""""""""""""""""""""""""
.. autosummary::
    :toctree: generated
    :template: custom.rst

    xpipe.xhandle.shearops.AutoCalibrateProfile


Input file manipulation
-----------------------

The bulk of the action is performed by the main writer class:

.. autosummary::
    :toctree: generated
    :template: custom.rst

    xpipe.xhandle.parbins.XIO

Additional helper functions are defined below.


Specify observational fields for the lens catalog:

.. autosummary::
    :toctree: generated

    xpipe.xhandle.parbins.get_file_lists
    xpipe.xhandle.parbins.field_cut
    xpipe.xhandle.parbins.get_fields_auto


Define K-means and Jackknife regions on the sphere:

.. autosummary::
    :toctree: generated

    xpipe.xhandle.parbins.assign_kmeans_labels
    xpipe.xhandle.parbins.assign_jk_labels
    xpipe.xhandle.parbins.extract_jk_labels


Load and prepare lens and random point catalogs:

.. autosummary::
    :toctree: generated

    xpipe.xhandle.parbins.load_lenscat
    xpipe.xhandle.parbins.prepare_lenses
    xpipe.xhandle.parbins.load_randcat
    xpipe.xhandle.parbins.prepare_random

------------



XSHEAR wrapper
---------------

metacal file tags
"""""""""""""""""

in addition the metacalibration tags are defined in
:py:data:`xpipe.xhandle.xwrap.sheared_tags` ::

    sheared_tags = ["_1p", "_1m", "_2p", "_2m"]

xshear config file writer
"""""""""""""""""""""""""

the main writer functions:

.. autosummary::
    :toctree: generated

    xpipe.xhandle.xwrap.write_xconf
    xpipe.xhandle.xwrap.write_custom_xconf

addittional helper functions:

.. autosummary::
    :toctree: generated

    xpipe.xhandle.xwrap.get_main_source_settings
    xpipe.xhandle.xwrap.get_main_source_settings_nopairs

    xpipe.xhandle.xwrap.addlines
    xpipe.xhandle.xwrap.get_pairlog
    xpipe.xhandle.xwrap.get_redges
    xpipe.xhandle.xwrap.get_shear
    xpipe.xhandle.xwrap.get_head
    xpipe.xhandle.xwrap.get_metanames


Running xshear
""""""""""""""

.. autosummary::
    :toctree: generated

    xpipe.xhandle.xwrap.create_infodict
    xpipe.xhandle.xwrap.call_xshear
    xpipe.xhandle.xwrap.call_chunks
    xpipe.xhandle.xwrap.multi_xrun


Random rotations of the source catalog
""""""""""""""""""""""""""""""""""""""

.. autosummary::
    :toctree: generated

    xpipe.xhandle.xwrap.single_rotate
    xpipe.xhandle.xwrap.serial_rotate
    xpipe.xhandle.xwrap.chunkwise_rotate

The catalog rotator object

.. autosummary::
    :toctree: generated
    :template: custom.rst

    xpipe.xhandle.xwrap.CatRotator

additional functions:

.. autosummary::
    :toctree: generated

    xpipe.xhandle.xwrap.get_rot_seeds
    xpipe.xhandle.xwrap.rot2d

------------


Postprocessing XSHEAR output
----------------------------

High level wrapper for postprocessing single parameter bins:

.. autosummary::
    :toctree: generated

    xpipe.xhandle.shearops.process_profile

Which wraps the main container class, responsible for most of the postprocessing:

.. autosummary::
    :toctree: generated
    :template: custom.rst

    xpipe.xhandle.shearops.StackedProfileContainer

Some other useful functions
"""""""""""""""""""""""""""

Extract **area-weighted** radial bins centers for the lensing measurement:

.. autosummary::
    :toctree: generated

    xpipe.xhandle.shearops.redges

Jackknife covariance between different parameter bins:

.. autosummary::
    :toctree: generated

    xpipe.xhandle.shearops.stacked_pcov

XSHEAR results I/O
""""""""""""""""""

The main reader function

.. autosummary::
    :toctree: generated

    xpipe.xhandle.ioshear.xread


Addtitional helpers for I/O:

.. autosummary::
    :toctree: generated

    xpipe.xhandle.ioshear.read_single_bin
    xpipe.xhandle.ioshear.read_multiple_bin


.. autosummary::
    :toctree: generated

    xpipe.xhandle.ioshear.xpatches
    xpipe.xhandle.ioshear.read_raw
    xpipe.xhandle.ioshear.read_multiple_raw
    xpipe.xhandle.ioshear.read_sheared_raw
    xpipe.xhandle.ioshear.read_multiple_sheared_raw

.. autosummary::
    :toctree: generated

    xpipe.xhandle.ioshear.makecat
    xpipe.xhandle.ioshear.read_lens_pos

------------

Cluster member contamination estimates
--------------------------------------

Tool to package information about what to do:

.. autosummary::
    :toctree: generated

    xpipe.xhandle.pzboost.create_infodicts

Calculate average photo-z P(z) PDF:
"""""""""""""""""""""""""""""""""""

.. autosummary::
    :toctree: generated

    xpipe.xhandle.pzboost.multi_pwsum_run
    xpipe.xhandle.pzboost.extract_pwsum
    xpipe.xhandle.pzboost.calc_pwsum

Additional useful tools:

.. autosummary::
    :toctree: generated

    xpipe.xhandle.pzboost.balance_infodicts
    xpipe.xhandle.pzboost.partition_tasks
    xpipe.xhandle.pzboost.call_pwsum_chunk

P(z) and Boost container object
"""""""""""""""""""""""""""""""

The Main Container Object is:

.. autosummary::
    :toctree: generated
    :template: custom.rst

    xpipe.xhandle.pzboost.PDFContainer


The JK-region collation is performed by:

.. autosummary::
    :toctree: generated

    xpipe.xhandle.pzboost.combine_pwsums

Classes for the P(z) decomposition:

The JK-region collation is performed by:

.. autosummary::
    :toctree: generated
    :template: custom.rst

    xpipe.xhandle.pzboost.BoostMixer
    xpipe.xhandle.pzboost.BoostMixerRandRef


Additional useful tools:

.. autosummary::
    :toctree: generated


    xpipe.xhandle.pzboost.check_pwsum_files
    xpipe.xhandle.pzboost.gauss
    xpipe.xhandle.pzboost.get_hist_zarr


------------


Useful tools
---------------

tools.catalogs
""""""""""""""

.. autosummary::
    :toctree: generated

    xpipe.tools.catalogs.to_pandas

additional functions:

.. autosummary::
    :toctree: generated

    xpipe.tools.catalogs.flat_type
    xpipe.tools.catalogs.flat_copy


tools.selector
""""""""""""""

.. automodule:: xpipe.tools.selector

.. currentmodule:: xpipe

.. autosummary::
    :toctree: generated

    xpipe.tools.selector.selector
    xpipe.tools.selector.matchdd
    xpipe.tools.selector.partition
    xpipe.tools.selector.safedivide


# TODO
Visualization

example function
""""""""""""""""

.. autosummary::
    :toctree: generated

    xpipe.xhandle.shearops.olivers_mock_function

