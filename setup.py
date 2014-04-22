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
      packages=find_packages(),
      scripts=[],
      zip_safe=False,
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'six',
          ],
      extras_require={
          'jit' : 'numba',
          'fast_fft' : 'pyFFTW',          
          }
      )
