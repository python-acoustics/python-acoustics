from setuptools import setup, find_packages

setup(
      name='acoustics',
      version='0.0',
      description="Acoustics module for Python.",
      long_description=open('README.md').read(),
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
