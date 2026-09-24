#!/usr/bin/env bash
# Rebuild everything and package the BeamNG mod zip into dist/.
set -euo pipefail
cd "$(dirname "$0")"

python3 tools/build_truck.py          # procedural geometry, textures, materials
python3 tools/build_jbeam.py          # physics + configs
# Blender: bevel/smooth, export final .dae + .glb, render selector thumbnails
python3 tools/blender_build.py --no-render --thumbs --samples 64
python3 tools/validate.py
python3 tools/simcheck.py

mkdir -p dist
rm -f dist/bobs_truck.zip
(cd mod && zip -qr ../dist/bobs_truck.zip vehicles)
echo "packaged dist/bobs_truck.zip"
