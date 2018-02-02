
.. toctree::
   :maxdepth: 2
   :hidden:


Python wrapper for xshear
============================

Select lenses by field
----------------------

.. autofunction:: proclens.xhandle.parbins.field_cut

Load catalogs
------------------

.. autofunction:: proclens.xhandle.parbins.load_lenscat

.. autofunction:: proclens.xhandle.parbins.load_randcat


Create xhsear-style lens catalog
--------------------------------

.. autofunction:: proclens.xhandle.ioshear.makecat

Read (RA, DEC) positions for lenses
-----------------------------------

.. autofunction:: proclens.xhandle.ioshear.read_lens_pos


Processing results from xshear
===================================


Read xshear output from single file
-----------------------------------

.. autofunction:: proclens.xhandle.ioshear.read_single_bin

Read xshear output from multiple files
--------------------------------------

.. autofunction:: proclens.xhandle.ioshear.read_multiple_bin


Base xshear readers
--------------------------------------

.. autofunction:: proclens.xhandle.ioshear.xread


.. autofunction:: proclens.xhandle.ioshear.xpatches


Logarithmic bin properties
--------------------------

.. autofunction:: proclens.xhandle.shearops.redges


Jackknife label assignment
--------------------------

.. autofunction:: proclens.xhandle.shearops.assign_jk_labels


Stacked Profile Container
-------------------------

.. autoclass:: proclens.xhandle.shearops.StackedProfileContainer

    .. automethod:: proclens.xhandle.shearops.StackedProfileContainer.prof_maker

    .. automethod:: proclens.xhandle.shearops.StackedProfileContainer.composite

    .. automethod:: proclens.xhandle.shearops.StackedProfileContainer.multiply

    .. automethod:: proclens.xhandle.shearops.StackedProfileContainer.drop_data

    .. automethod:: proclens.xhandle.shearops.StackedProfileContainer.to_sub_dict

    .. automethod:: proclens.xhandle.shearops.StackedProfileContainer.from_sub_dict


Jackknife Covariance estimate from multiple lensing profiles
------------------------------------------------------------

.. autofunction:: proclens.xhandle.shearops.stacked_pcov



Ancillary utilities
=====================


Observed to absolute magnitude conversion
----------------------------------------------------------

.. autoclass:: proclens.tools.magtools.AbsMagConverter

.. au

FITS to pandas conversion
--------------------------

.. autofunction:: proclens.tools.catalogs.to_pandas


parameter selection
--------------------

.. autofunction:: proclens.tools.selector.selector


list partitioning
-----------------

.. autofunction:: proclens.tools.selector.partition


safe divide
-----------

.. autofunction:: proclens.tools.selector.safedivide

Visualization
=============

corner plot
-----------

.. autofunction:: proclens.tools.visual.corner

