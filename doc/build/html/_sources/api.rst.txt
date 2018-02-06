
.. toctree::
   :maxdepth: 2
   :hidden:


Python wrapper for xshear
============================

Select lenses by field
----------------------

.. autofunction:: xpipe.xhandle.parbins.field_cut

.. autofunction:: xpipe.xhandle.parbins.get_fields_auto

Load catalogs
------------------

.. autofunction:: xpipe.xhandle.parbins.load_lenscat

.. autofunction:: xpipe.xhandle.parbins.load_randcat

.. autofunction:: xpipe.xhandle.parbins.prepare_lenses

.. autofunction:: xpipe.xhandle.parbins.prepare_random

Create xshear-style input files
--------------------------------

.. autoclass:: xpipe.xhandle.parbins.XIO
    :members:


Create xhsear-style lens catalog
--------------------------------

.. autofunction:: xpipe.xhandle.ioshear.makecat

Read (RA, DEC) positions for lenses
-----------------------------------

.. autofunction:: xpipe.xhandle.ioshear.read_lens_pos


Processing results from xshear
===================================


Read xshear output from single file
-----------------------------------

.. autofunction:: xpipe.xhandle.ioshear.read_single_bin

Read xshear output from multiple files
--------------------------------------

.. autofunction:: xpipe.xhandle.ioshear.read_multiple_bin


Base xshear readers
--------------------------------------

.. autofunction:: xpipe.xhandle.ioshear.xread


.. autofunction:: xpipe.xhandle.ioshear.xpatches


Logarithmic bin properties
--------------------------

.. autofunction:: xpipe.xhandle.shearops.redges


Jackknife label assignment
--------------------------

.. autofunction:: xpipe.xhandle.parbins.assign_kmeans_labels

.. autofunction:: xpipe.xhandle.parbins.assign_jk_labels

.. autofunction:: xpipe.xhandle.parbins.extract_jk_labels


Stacked Profile Container
-------------------------

.. autoclass:: xpipe.xhandle.shearops.StackedProfileContainer

    .. automethod:: xpipe.xhandle.shearops.StackedProfileContainer.prof_maker

    .. automethod:: xpipe.xhandle.shearops.StackedProfileContainer.composite

    .. automethod:: xpipe.xhandle.shearops.StackedProfileContainer.multiply

    .. automethod:: xpipe.xhandle.shearops.StackedProfileContainer.drop_data

    .. automethod:: xpipe.xhandle.shearops.StackedProfileContainer.to_sub_dict

    .. automethod:: xpipe.xhandle.shearops.StackedProfileContainer.from_sub_dict


Jackknife Covariance estimate from multiple lensing profiles
------------------------------------------------------------

.. autofunction:: xpipe.xhandle.shearops.stacked_pcov



Ancillary utilities
=====================


Observed to absolute magnitude conversion
----------------------------------------------------------

.. autoclass:: xpipe.tools.magtools.AbsMagConverter


FITS to pandas conversion
--------------------------

.. autofunction:: xpipe.tools.catalogs.to_pandas


parameter selection
--------------------

.. autofunction:: xpipe.tools.selector.selector


list partitioning
-----------------

.. autofunction:: xpipe.tools.selector.partition


safe divide
-----------

.. autofunction:: xpipe.tools.selector.safedivide


distribution matching
---------------------

.. autofunction:: xpipe.tools.selector.matchdd

Visualization
=============

corner plot
-----------

.. autofunction:: xpipe.tools.visual.corner

