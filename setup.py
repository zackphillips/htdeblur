from setuptools import setup, find_packages
import os, sys
import subprocess

# Define version
__version__ = 0.02

setup( name             = 'htdeblur'
     , version          = __version__
     , description      = 'Python implementation of high-throughput imaging using motion deblur'
     , license          = 'BSD'
     , packages         = find_packages()
     , include_package_data = True
     , install_requires = ['sympy', 'numexpr', 'contexttimer', 'imageio', 'matplotlib_scalebar', 'tifffile', 'pyserial', 'numpy', 'scipy', 'scikit-image', 'planar']
     )
