# Nix function for building this package
{ lib
, buildPythonPackage
, flit
, flit-core
, wheel
, pytest
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
, ensureNewerSourcesForZipFilesHook
, development ? false
}:

let
  sdist = stdenv.mkDerivation {
    name = "acoustics-sdist";
    src = ./.;

    nativeBuildInputs = [
      python
      flit # Use flit front-end here because we don't have a sdist hook
      wheel
      # Ensure files are after 1980 so users not using
      # Nix and buildPythonPackage can built a wheel as well.
      ensureNewerSourcesForZipFilesHook
    ];

    buildPhase = ":";

    installPhase = ''
      flit build --format sdist
      mkdir -p $out
      cp dist/* $out/
    '';

    strictDeps = true;
  };

in buildPythonPackage rec {
  pname = "acoustics";
  version = "0.2.6";
  format = "pyproject";

  src = "${sdist}/${pname}*";

  nativeBuildInputs = [
    flit-core
  ] ++ lib.optionals development [ sphinx pylint yapf ];
  propagatedBuildInputs = [ numpy scipy matplotlib pandas six tabulate ];

  nativeCheckInputs = [
    pytest
  ];

  meta = {
    description = "Acoustics module for Python";
  };

  checkPhase = ''
    pushd tests
    LC_ALL="en_US.UTF-8" pytest .
    popd
  '';

  passthru.sdist = sdist;
}
