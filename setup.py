from setuptools import setup, find_packages

setup(name="proclens",
      packages=find_packages(),
      description="Measurement and data processing for DES Y1 cluster lensing",
      install_requires=['numpy', 'scipy', 'pandas', 'astropy', 'fitsio', 'kmeans_radec', ],
      author="Tamas Norbert Varga",
      author_email="T.Varga@usm.lmu.de",
      version="0.1")