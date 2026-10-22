# GS-AgriField baseline benchmark

Scripts and configurations of the baseline benchmark of the GS-AgriField paper: four monocular
front ends of pySLAM (ORB, AKAZE, SIFT+LightGlue, SuperPoint+LightGlue) on the eleven sequences of
the dataset, with a retry-until-success protocol.

This branch is pySLAM at upstream commit `b747a10` plus the changes used for the benchmark
(see the commits of the branch `gs-agrifield-benchmark`). Default parameter values of pySLAM are
unchanged; the benchmark only sets the presets below.

## Contents

| File | Role |
|---|---|
| `sequences.tsv` | the eleven sequences, numbered as in the paper, with their MARV bag names |
| `setup.py` | converts the bags (optional), prepares the pySLAM folder datasets and writes one config per run in `runs/` |
| `run_all.sh` | runs every (sequence, method) pair with the retry-until-success protocol |
| `run_docker.sh` | runs one config in the Docker image |
| `evaluate.py` | evaluates every run against the RTK-GNSS/INS reference (`results/runs.csv`) |
| `aggregate.py` | applies the protocol and writes the per-pair results and tables (`results/per_pair.csv`) |
| `capture_versions.sh` | records the software versions in `versions/` |
| `settings/` | camera settings of rig A and rig B at image scale 0.4 |
| `results/` | reference results of the paper (54 runs, 44 pairs) |
| `versions/reference_2026-10-17/` | environment of the runs of the paper |

## Benchmark design

| Item | Value |
|---|---|
| Sensor mode | monocular, 10 Hz, one thread for the numerical libraries |
| Image scale | 0.4 (1152 x 744 pixels), intrinsics scaled accordingly |
| Front ends (pySLAM names) | `ORB2`, `AKAZE`, `LIGHTGLUESIFT`, `LIGHTGLUE` (SuperPoint) |
| Features per image | 8000 |
| Loop closing | enabled, `DBOW3_INDEPENDENT` |
| Attempts | up to 3 per pair, seeds 12345, 23456, 34567; stop at the first success |
| Success | sequence completed and at least 90 % of the frames tracked |
| Early stop | 50 consecutive lost or relocalizing frames |
| Evaluation | evo 1.31.1; association with max 20 ms; Sim(3) Umeyama alignment; reference in the camera frame |
| KITTI drift | sub-trajectories of 10 to 50 m, started every 10 frames |
| Scale drift | coefficient of variation of the estimated-to-reference distance ratio over consecutive 10 m segments; segments with less than 5 m of net displacement skipped |

The reported run of a pair is its first successful attempt, never the most accurate one.
pySLAM is not deterministic (tracking, local mapping and loop closing run in parallel threads),
so a new run of a pair can differ slightly from the reference results.

## Data

Download the bags from MARV with the tools of the dataset repository
(https://github.com/itsayoushka/agri-mono-vo-dataset) and arrange them as:

```
$GSAF_DATA/bags/<bag>/                 raw ROS 2 bags (metadata.yaml + .mcap files)
$GSAF_DATA/kitti/sequences/<bag>/      KITTI-like conversions
```

`setup.py` converts the missing sequences itself if `GSAF_CONVERTER` points to
`scripts/mono_camera_kitti_converter.py` of the dataset repository and `CONVERTER_IMAGE` to a
Docker image with ROS 2 and the MCAP storage plugin (see the dataset repository). Conversions made
beforehand are used as they are.

## Environment

The runs of the paper used a Docker image built from the `ubuntu24_cuda` image of
[rosdocker](https://github.com/luigifreda/rosdocker) (CUDA 12.8, Ubuntu 24.04), with pySLAM installed
in `.venv/` at the root of this repository by `install_all.sh` (Python 3.12). Versions are in
`versions/reference_2026-10-17/` (torch 2.13.0+cu126, OpenCV 4.11.0, NVIDIA driver 535,
RTX 4070 Laptop GPU).

`environment.txt` names the local commit `8f41459` on which the runs were made: it is upstream
`b747a10` plus two commits that only add configuration files of other experiments, not used here.
The local changes it counts are those of this branch, plus configuration files of other
experiments that the benchmark does not use.

1. Build the `ubuntu24_cuda` image of rosdocker and start a container with this repository mounted.
2. In the container, run `./install_all.sh` at the root of this repository, then build the C++ core
   (`./build_cpp_core.sh`). Commit the container as an image, e.g. `pyslam-gsaf:latest`.
3. On the host, create a Python environment for the evaluation:
   `python3 -m venv .venv_eval && .venv_eval/bin/pip install evo==1.31.1 numpy pandas scipy matplotlib`.

## Running

```
cd configs/gs_agrifield_benchmark
export GSAF_DATA=/path/to/data PYSLAM_IMAGE=pyslam-gsaf:latest
python3 setup.py --no-convert           # or without --no-convert to convert missing bags
./run_all.sh                            # all pairs; ./run_all.sh __orb2 for one method
.venv_eval/bin/python evaluate.py       # results/runs.csv
.venv_eval/bin/python aggregate.py      # results/per_pair.csv and tables
```

`run_all.sh` can be interrupted and run again: finished attempts are skipped (`logs/`).
The 54 runs of the paper took about 9 h in total on the hardware of the paper.

## Environment variables

| Variable | Default | Use |
|---|---|---|
| `GSAF_DATA` | required | data root (`bags/`, `kitti/sequences/`) |
| `PYSLAM_IMAGE` | `pyslam-gsaf:latest` | Docker image with the pySLAM environment |
| `PYSLAM_ROOT` | root of this repository | pySLAM tree used by the scripts |
| `EVAL_PYTHON` | `.venv_eval/bin/python` | Python with evo for the protocol check in `run_all.sh` |
| `STOP_LOST` | `50` | consecutive lost frames before a run is stopped |
| `GSAF_CONVERTER`, `CONVERTER_IMAGE` | unset, `jazzy-vo-converter:latest` | bag conversion in `setup.py` |

## License

pySLAM and this branch are distributed under the GPLv3 license (see `LICENSE` at the root).
