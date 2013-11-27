from setuptools import setup

setup(
      name='acoustics',
      version='0.0',
      description="Acoustics module for Python.",
      long_description=open('README.md').read(),
      author='Python Acoustics',
      author_email='fridh@fridh.nl',
      license='LICENSE',
      packages=['acoustics'],
      scripts=[],
      zip_safe=False,
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'mayavi',
          ],
      )
