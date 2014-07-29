import os
from setuptools import setup, find_packages

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
      packages=find_packages(exclude=["tests"]),
      scripts=[],
      zip_safe=False,
      install_requires=[
          'numpy >=1.8',
          'scipy >= 0.13',
          'matplotlib',
          'six >= 1.4.1',
          ],
      extras_require={
          'jit': 'numba',
          'fast_fft': 'pyFFTW',
          }
      )
