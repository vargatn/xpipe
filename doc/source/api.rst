
.. toctree::
   :maxdepth: 2
   :hidden:

Python wrapper for xshear
============================

Create xhsear-style lens catalog
--------------------------------

.. autofunction:: proclens.xhandle.ioshear.makecat

Read (RA, DEC) positions for lenses
-----------------------------------

.. autofunction:: proclens.xhandle.ioshear.read_lens_pos


Processing results from xshear
===================================




Read xshear output
------------------

.. autofunction:: proclens.xhandle.ioshear.read_single_bin


.. autofunction:: proclens.xhandle.ioshear.read_multiple_bin


.. autofunction:: proclens.xhandle.ioshear.xread


.. autofunction:: proclens.xhandle.ioshear.xpatches




Ancillary utilities
=====================

Observed to absolute magnitude conversion
----------------------------------------------------------

.. autoclass:: proclens.tools.magtools.AbsMagConverter
    :members:


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

