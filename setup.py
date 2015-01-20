import os
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
from setuptools.extension import Extension
import sys

import numpy as np

if os.path.exists('README.md'):
    long_description = open('README.md').read()
else:
    long_description = "A Python library aimed at acousticians."


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)

setup(
      name='acoustics',
      version='0.0',
      description="Acoustics module for Python.",
      long_description=long_description,
      author='Python Acoustics',
      author_email='fridh@fridh.nl',
      license='LICENSE',
      packages=find_packages(exclude=["tests"]),
      #py_modules=['turbulence'],
      scripts=[],
      zip_safe=False,
      install_requires=[
          'numpy >=1.8',
          'scipy >= 0.14',
          'matplotlib',
          'six >= 1.4.1',
          'numexpr',
          ],
      extras_require={
          'jit': 'numba',
          'fast_fft': 'pyFFTW',
          'io': 'pandas',
          },
      tests_require=['pytest'],
      cmdclass={'test': PyTest},
      ext_modules=[Extension('acoustics._signal', ['acoustics/_signal.c'])],
      include_dirs=[np.get_include()]
      )
