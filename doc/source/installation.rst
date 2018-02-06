

Installation
=============

The package itself can be obtained from bitbucket: ::

    git clone https://vargatn@bitbucket.org/vargatn/xpipe.git

There are two modes: *pipeline* and *API*, for instructions, see the below sections.


**Requirements and Dependencies**

The code is written in :code:`python 2.7`.

Additinal required packages:

* *Anaconda*: :code:`numpy`, :code:`scipy`, :code:`pandas`, :code:`astropy`,

* *pip* :code:`fitsio`

* *manual install*: kmeans_radec_

.. _kmeans_radec: https://github.com/esheldon/kmeans_radec

In addition *xshear* requires a C99 compliant compiler.


Pipeline mode
---------------

First build the pipeline by executing ::

    make pipeline

In the main folder of the repository. This performs the following steps:


* Sets up the main *xpipe* package

* Pulls and builds the submodule *xshear*. The executable is located at::

        [XPIPE_FOLDER]/submodules/xshear/bin/xshear

* Writes a logfile to the user path. This is necessary for the
  package to find the absolute location of the config files.

After this the package can be simply imported as ::

   import xpipe

and ther are some pre-defined scripts located at ::

    [XPIPE_FOLDER]/bin/redpipe

a detaild description of what they do is given in the :doc:`Pipeline Guide <pipeline_guide>` section.


When using these scripts, be sure to note the config files located within ::

    [XPIPE_FOLDER]/settings/

namely ``default_params.yml`` and ``default_inputs.yml``, as these will define what happens when you execute them.
You can define your own settings in ``params.yml`` and ``inputs.yml``,
which are automatically looked for when using the pipeline.
(more on how to specify these settings is explained in the  :doc:`Config files explained <config>` page.


API mode
------------

The *API* mode can be accessed by installing the python package: ::

    python setup.py install

Then simply import it in a python session: ::

   import xpipe

Note that this only gives access to the python part of the code, you have to compile *xshear* manually, and keep track
of the file paths.



