#!/bin/sh
echo "Building twine"
nix-build -A python3.pkgs.twine -o twine release.nix
echo "Uploading with twine"
twine/bin/twine upload dist/*
echo "Successfully uploaded"