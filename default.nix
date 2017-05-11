# Nix function for building this package
# To always get the latest version you could use
#
# acoustics = (import fetchTarball {
#   url = https://github.com/python-acoustics/python-acoustics/archive/master.tar.gz;
# });
#
# Note that the above should still be called, with the following arguments.
{ buildPythonPackage
, pytest
, cython
, cytoolz
, numpy
, scipy
, matplotlib
, pandas
, six
, tabulate
}:

buildPythonPackage rec {
  name = "acoustics-${version}";
  version = "0.1.2dev";

  src = ./.;

  preBuild = ''
    make clean
  '';

  buildInputs = [ pytest cython ];
  propagatedBuildInputs = [ cytoolz numpy scipy matplotlib pandas six tabulate ];

  meta = {
    description = "Acoustics module for Python";
  };

  doCheck = false;
}
