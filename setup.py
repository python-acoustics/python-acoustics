import os
from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy as np

if os.path.exists('README.md'):
    long_description = open('README.md').read()
else:
    long_description = "A Python library aimed at acousticians."

setup(
      name='acoustics',
      version='0.0',
      description="Acoustics module for Python.",
      long_description=long_description,
      author='Python Acoustics',
      author_email='fridh@fridh.nl',
      license='LICENSE',
      #packages=find_packages(exclude=["tests"]),
      py_modules=['turbulence'],
      scripts=[],
      zip_safe=False,
      install_requires=[
          'numpy >=1.8',
          'scipy >= 0.13',
          'matplotlib',
          'six >= 1.4.1',
          'cython',
          'numexpr',
          ],
      extras_require={
          'jit': 'numba',
          'fast_fft': 'pyFFTW',
          'io:pandas',
          },
      ext_modules = cythonize('acoustics/*.pyx'),
      include_dirs=[np.get_include()]
      )
