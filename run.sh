#!/usr/bin/env bash
# Reproduce el experimento completo con un único comando:
#     ./run.sh                      # mapa semántico (SegFormer)
#     ./run.sh mode:=color          # paso 1: nube coloreada con la imagen
#     ./run.sh backend:=yolo        # segmentación con YOLO-seg
# Cualquier argumento se pasa tal cual al launch. Resultados en ./output
set -euo pipefail
cd "$(dirname "$0")"

export BAG_DIR="${BAG_DIR:-$(realpath ../1710255615_2024-03-12-15-00-16)}"
if [ ! -f "$BAG_DIR/metadata.yaml" ]; then
  echo "No encuentro el bag en $BAG_DIR. Exporta BAG_DIR=/ruta/al/bag" >&2
  exit 1
fi
mkdir -p output

if ! docker image inspect ri-jazzy-base >/dev/null 2>&1; then
  echo ">> Construyendo la imagen base de la asignatura (solo la primera vez)"
  docker build -t ri-jazzy-base -f docker/Dockerfile.base docker
fi
xhost +local: >/dev/null 2>&1 || true
docker compose build
docker compose run --rm semantic_mapping \
  ros2 launch semantic_mapping_bringup semantic_mapping.launch.py "$@"
