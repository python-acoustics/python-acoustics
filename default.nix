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
, glibcLocales
}:

buildPythonPackage rec {
  pname = "acoustics";
  version = "dev";

  src = ./.;

  preBuild = ''
    make clean
  '';

  checkInputs = [ pytest glibcLocales ];
  buildInputs = [ cython ];
  propagatedBuildInputs = [ cytoolz numpy scipy matplotlib pandas six tabulate ];

  meta = {
    description = "Acoustics module for Python";
  };

  checkPhase = ''
    LC_ALL="en_US.UTF-8"
    py.test tests
  '';

}
