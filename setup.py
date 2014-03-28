import os
from setuptools import setup, find_packages

long_description = "A Python library aimed at acousticians."
if os.path.exists('README.md'):
    long_description=open('README.md').read()

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
          ],
      )
