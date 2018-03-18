

=============
API Reference
=============


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
