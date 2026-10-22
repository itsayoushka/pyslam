#!/usr/bin/env bash
# Record everything needed to reproduce the benchmark into versions/ (run once before the experiments).
set -u
B=$(cd "$(dirname "$0")" && pwd)
W=${PYSLAM_ROOT:-$(cd "$B/../.." && pwd)}
IMAGE=${PYSLAM_IMAGE:-pyslam-gsaf:latest}
PY=${EVAL_PYTHON:-$B/.venv_eval/bin/python}
V=$B/versions
mkdir -p "$V"
{
  echo "date: $(date -Iseconds)"
  echo "host: $(uname -srm)"
  echo "nvidia_driver: $(nvidia-smi --query-gpu=driver_version,name --format=csv,noheader 2>/dev/null)"
  echo "docker_image: $IMAGE $(docker image inspect $IMAGE --format '{{.Id}}')"
  echo "converter_image: ${CONVERTER_IMAGE:-jazzy-vo-converter:latest} $(docker image inspect ${CONVERTER_IMAGE:-jazzy-vo-converter:latest} --format '{{.Id}}' 2>/dev/null)"
  echo "pyslam_upstream: https://github.com/luigifreda/pyslam"
  echo "pyslam_commit: $(git -C $W rev-parse HEAD)"
  echo "pyslam_local_changes: $(git -C $W diff --stat | tail -1)   (full patch: pyslam_local_changes.patch)"
  echo "converter_sha256: $( [ -n "${GSAF_CONVERTER:-}" ] && sha256sum "$GSAF_CONVERTER" | cut -c1-64)"
  echo "evo: $($PY -c 'import evo;print(evo.__version__)')"
} > "$V/environment.txt"
git -C $W diff > "$V/pyslam_local_changes.patch"
sha256sum $W/thirdparty/superpoint/superpoint_v1.pth $W/thirdparty/accelerated_features/weights/xfeat.pt \
  $(find "$B/torch_cache" -name '*.pth' 2>/dev/null) > "$V/model_weights_sha256.txt" 2>/dev/null
docker run --rm --gpus all --runtime=nvidia -e HOME=/tmp -v "$W:$W" $IMAGE \
  bash -c "source $W/.venv/bin/activate && pip freeze && python -c 'import torch,cv2;print(\"#torch\",torch.__version__,\"cuda_available\",torch.cuda.is_available(),\"#opencv\",cv2.__version__)'" \
  > "$V/pip_freeze_container.txt" 2>/dev/null
cp "$B/setup.py" "$B/sequences.tsv" "$V/"
echo "versions written to $V"
