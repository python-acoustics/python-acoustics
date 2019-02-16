import pkg_resources
# We use setup.cfg for which we need a 'recent' setuptools version
pkg_resources.require('setuptools>=30.3.0')

import setuptools
from setuptools.command.test import test as TestCommand
import sys


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
)
