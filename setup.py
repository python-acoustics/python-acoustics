import os
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import sys

import numpy as np
from Cython.Build import cythonize

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
      version='0.1.2',
      description="Acoustics module for Python.",
      long_description=long_description,
      author='Python Acoustics',
      author_email='fridh@fridh.nl',
      license='LICENSE',
      packages=find_packages(exclude=["tests"]),
      package_dir={'acoustics': 'acoustics'},
      package_data={'acoustics': ['data/*']},
      #py_modules=['turbulence'],
      scripts=[],
      zip_safe=False,
      install_requires=[
          'numpy >=1.8',
          'scipy >= 0.16',
          'matplotlib',
          'six >= 1.4.1',
          'cython',
          'pandas>=0.15',
          'tabulate',
          ],
      extras_require={
          'documentation': 'sphinx',
          'jit': 'numba',
          'fast_fft': 'pyFFTW',
          },
      tests_require=['pytest'],
      cmdclass={'test': PyTest},
      ext_modules=cythonize('acoustics/*.pyx'),
      include_dirs=[np.get_include()]
      )
