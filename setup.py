from setuptools import setup, find_packages

setup(name="proclens",
      packages=find_packages(),
      description="Data processing preparation and analysis functions for DES Y1 cluster lensing",
      install_requires=['numpy', 'scipy', 'pandas', 'astropy', 'fitsio', 'kmeans_radec', ],
      author="Tamas Norbert Varga",
      author_email="vargat@usm.lmu.de",
      version="0.1")