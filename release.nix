{ nixpkgs ? (fetchTarball "channel:nixos-18.09")
}:

let
  overrides = self: super: {
    acoustics = self.callPackage ./. {};
  };

  overlay = self: super: {
    python36 = super.python36.override{packageOverrides=overrides;};
    python37 = super.python37.override{packageOverrides=overrides;};
  };

in import nixpkgs {
  overlays = [ overlay ];
}

