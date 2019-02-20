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
, bootstrapped-pip
, stdenv
, python
, development ? false
}:

let
  sdist = stdenv.mkDerivation {
    name = "acoustics-sdist";
    src = ./.;

    nativeBuildInputs = [
      python bootstrapped-pip
    ];

    buildPhase = ":";

    installPhase = ''
      ${python.interpreter} setup.py sdist
      mkdir -p $out
      cp dist/* $out/
    '';
  };

in buildPythonPackage rec {
  pname = "acoustics";
  version = "0.2.0.post2";

  src = "${sdist}/${pname}*";

  checkInputs = [ pytest glibcLocales ];
  nativeBuildInputs = lib.optionals development [ sphinx pylint yapf ];
  propagatedBuildInputs = [ cytoolz numpy scipy matplotlib pandas six tabulate ];

  meta = {
    description = "Acoustics module for Python";
  };

  checkPhase = ''
    pushd tests
    LC_ALL="en_US.UTF-8" py.test .
    popd
  '';

  passthru.sdist = sdist;
}
