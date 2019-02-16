{ nixpkgs ? (fetchTarball "channel:nixos-18.09")
}:

with nixpkgs;

let

  # Create an sdist given a derivation.
  # Should add an sdist and wheel output to buildPythonPackage
  create-sdist = drv:
    drv.overridePythonAttrs(oldAttrs: with oldAttrs; rec {
      name = "${pname}-${version}-sdist";

      postBuild = ''
        rm -rf dist
        ${drv.pythonModule.interpreter} nix_run_setup sdist

      '';

      installPhase = ''
        mkdir -p $out
        mv dist/*.tar.gz $out/
      '';

      fixupPhase = "true";
      doCheck = false;
      propagatedBuildInputs = [];
    });

  overrides = self: super: {
    acoustics = super.callPackage ./default.nix {
      development = true;
    };
    acoustics-sdist = create-sdist (self.acoustics.override {
      development = false;
    });
  };

  overlay = self: super: {
    python36 = super.python36.override{packageOverrides=overrides;};
    python37 = super.python37.override{packageOverrides=overrides;};
  };

in import nixpkgs {
  overlays = [ overlay ];
}
