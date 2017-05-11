# Nix expression to build the package.
# Calling `nix-build develop.nix` will build a
# Python 3.5 version of the package.

let
  pkgs = import <nixpkgs> {};
  python = pkgs.python35;
#   python = "python35";
#   pythonPackages = pkgs.${python+"Packages"};

in pkgs.callPackage ./default.nix {
  inherit (python.pkgs) buildPythonPackage pytest cython cytoolz numpy scipy matplotlib pandas six tabulate;
}
