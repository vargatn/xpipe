from setuptools import setup, find_packages

setup(name="xpipe",
      packages=find_packages(),
      description="Measurement and data processing for DES Y1 cluster lensing",
      install_requires=['numpy', 'scipy', 'pandas', 'astropy', 'fitsio', 'kmeans_radec', ],
      dependency_links=['http://github.com/user/repo/tarball/master#egg=package-1.0'],
      author="Tamas Norbert Varga",
      author_email="T.Varga@physik.lmu.de",
      version="0.4")

