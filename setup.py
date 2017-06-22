import setuptools

if tuple([int(x) for x in setuptools.__version__.split('.')[:3]]) < (30,3,0):
    raise ValueError("This package requires setuptools 30.3.0 or newer.")

from setuptools.command.test import test as TestCommand
import sys
import numpy as np
from Cython.Build import cythonize


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)

setuptools.setup(
    cmdclass={'test': PyTest},
    ext_modules=cythonize('acoustics/*.pyx'),
    include_dirs=[np.get_include()]
)
