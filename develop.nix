# Nix expression to build the package.
# Calling `nix-build develop.nix` will build a
# Python 3.5 version of the package.

let
  pkgs = import <nixpkgs> {};
  python = pkgs.python3;

in pkgs.callPackage ./default.nix {
  inherit (python.pkgs) buildPythonPackage pytest cython cytoolz numpy scipy matplotlib pandas six tabulate;
}
