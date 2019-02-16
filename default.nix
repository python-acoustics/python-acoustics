# Nix function for building this package
# To always get the latest version you could use
#
# acoustics = (import fetchTarball {
#   url = https://github.com/python-acoustics/python-acoustics/archive/master.tar.gz;
# });
#
# Note that the above should still be called, with the following arguments.
{ lib
, buildPythonPackage
, pytest
, cytoolz
, numpy
, scipy
, matplotlib
, pandas
, six
, tabulate
, glibcLocales
, pylint
, yapf
, sphinx
, development ? false
}:

buildPythonPackage rec {
  pname = "acoustics";
  version = "0.2.0.post1";

  src = ./.;

  preBuild = ''
    make clean
  '';

  checkInputs = [ pytest glibcLocales ];
  nativeBuildInputs = lib.optionals development [ sphinx pylint yapf ];
  propagatedBuildInputs = [ cytoolz numpy scipy matplotlib pandas six tabulate ];

  meta = {
    description = "Acoustics module for Python";
  };

  checkPhase = ''
    LC_ALL="en_US.UTF-8" py.test tests
  '';
}
