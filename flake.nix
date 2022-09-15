{
  description = "Python module for loading sound files as xarray arrays.";

  inputs.nixpkgs.url = "nixpkgs/nixpkgs-unstable";
  inputs.utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, utils }: let

    attribute = "acoustics";

    inherit (nixpkgs) lib;

    interpreters-to-test = [
      "python39"
      "python310"
      "python3"
    ];

    # Overlay we expose externally
    overlay = final: prev: {
      pythonPackagesOverrides = (prev.pythonPackagesOverrides or []) ++ [
        (self: super: {
          "${attribute}" = self.callPackage ./. {};
        })
      ];
    };

    create-overlay-for-interpreter = interpreter: (self: super: {
      "${interpreter}" = let
        self = super.${interpreter}.override {
          inherit self;
          packageOverrides = lib.composeManyExtensions super.pythonPackagesOverrides;
        };
      in self;
    });

    overlay-for-testing = lib.composeManyExtensions ([
      overlay 
    ] ++ (map create-overlay-for-interpreter interpreters-to-test));

  in {
    overlays = {
      default = overlay;
    };
  } // (utils.lib.eachSystem [ "x86_64-linux" ] (system: let
    # Our own overlay does not get applied to nixpkgs because that would lead to
    # an infinite recursion. Therefore, we need to import nixpkgs and apply it ourselves.
    pkgs = import nixpkgs {
      inherit system;
      overlays = [
        overlay-for-testing
      ];
    };
    python = pkgs.python3;
  in rec {
    packages = rec {
      # Development environment that includes our package, its dependencies
      # and additional dev inputs.
      devEnv = python.withPackages(_: (pkg.nativeBuildInputs ++ pkg.propagatedBuildInputs));
      pkg = python.pkgs."${attribute}";
      default = pkg;
    } // (lib.genAttrs interpreters-to-test (interpreter: pkgs."${interpreter}".pkgs.acoustics));

    devShells = {
      default = pkgs.mkShell {
        nativeBuildInputs = [
          packages.devEnv
        ];
        shellHook = ''
          export PYTHONPATH=$(readlink -f .):$PYTHONPATH
        '';
      };
    };
    checks = lib.genAttrs interpreters-to-test (interpreter: pkgs."${interpreter}".pkgs.acoustics);
  }));
}
