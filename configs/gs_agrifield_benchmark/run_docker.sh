#!/usr/bin/env bash
# Run one pySLAM evaluation config in the benchmark Docker image, with a persistent TORCH_HOME so that the
# learned-model weights (SuperPoint, LightGlue) are downloaded once.
#   usage: run_docker.sh <config.json>
#   env:   PYSLAM_IMAGE  Docker image with the pySLAM environment (see README.md)
#          GSAF_DATA     data root, mounted at the same path in the container
#          STOP_LOST     early stop after this many consecutive lost frames (default 50)
B=$(cd "$(dirname "$0")" && pwd)
W=${PYSLAM_ROOT:-$(cd "$B/../.." && pwd)}
IMAGE=${PYSLAM_IMAGE:-pyslam-gsaf:latest}
: "${GSAF_DATA:?set GSAF_DATA to the data root}"
CACHE=$B/torch_cache
mkdir -p "$CACHE"
docker run --rm --net=host --ipc=host --shm-size=16g -u "$(id -u):$(id -g)" \
  --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all -e NVIDIA_VISIBLE_DEVICES=all --runtime=nvidia \
  -e HOME=/tmp -e TORCH_HOME="$CACHE" -e PYSLAM_STOP_AFTER_LOST_FRAMES="${STOP_LOST:-50}" -e PYSLAM_EXPORT_FRAME_LOGS=1 \
  -e MPLBACKEND=Agg -e QT_QPA_PLATFORM=offscreen \
  -v "$W:$W:rw" -v "$GSAF_DATA:$GSAF_DATA:ro" \
  -w "$W" "$IMAGE" \
  bash -c "source $W/.venv/bin/activate && python main_slam_evaluation.py -c $1"
