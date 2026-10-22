#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir (this should be the main folder directory of PLVS)
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SCRIPTS_DIR="$ROOT_DIR/scripts"
cd "$SCRIPT_DIR"

function make_dir(){
if [ ! -d $1 ]; then
    mkdir $1
fi
}

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    version=$(lsb_release -a 2>&1)  # ubuntu version
else 
    version=$OSTYPE
    echo "OS: $version"
fi

# Check if pixi is activated
if [[ -n "$PIXI_PROJECT_NAME" ]]; then
    PIXI_ACTIVATED=true
    echo "Pixi environment detected: $PIXI_PROJECT_NAME"

    source "$SCRIPTS_DIR/pixi_python_config.sh"
else
    PIXI_ACTIVATED=false
fi

# ====================================================
# check if we have external options
EXTERNAL_OPTIONS=$@
if [[ -n "$EXTERNAL_OPTIONS" ]]; then
    echo "external option: $EXTERNAL_OPTIONS" 
fi

# OpenCV_DIR="$SCRIPT_DIR/../opencv/install/lib/cmake/opencv4"
# if [[ -d "$OpenCV_DIR" ]]; then
#     EXTERNAL_OPTIONS="$EXTERNAL_OPTIONS -DOpenCV_DIR=$OpenCV_DIR"
# fi 

# check if WITH_MARCH_NATIVE is not set
if [[ -z "$WITH_MARCH_NATIVE" ]]; then
    WITH_MARCH_NATIVE=ON
    EXTERNAL_OPTIONS="$EXTERNAL_OPTIONS -DWITH_MARCH_NATIVE=$WITH_MARCH_NATIVE"
fi

EXTERNAL_OPTIONS+=" -DCMAKE_POLICY_VERSION_MINIMUM=3.5"

echo "EXTERNAL_OPTIONS: $EXTERNAL_OPTIONS"

# ====================================================

make_dir build
cd build
cmake .. $EXTERNAL_OPTIONS
# Keep build pressure low by default to avoid UI/system freeze on laptops.
if [[ -z "$PYSLAM_GTSAM_BUILD_JOBS" ]]; then
    BUILD_JOBS=2
else
    BUILD_JOBS="$PYSLAM_GTSAM_BUILD_JOBS"
fi
if [[ -z "$PYSLAM_GTSAM_BUILD_LOAD" ]]; then
    BUILD_LOAD=2
else
    BUILD_LOAD="$PYSLAM_GTSAM_BUILD_LOAD"
fi
echo "Using build parallelism: jobs=$BUILD_JOBS load_limit=$BUILD_LOAD"
make -j"$BUILD_JOBS" -l"$BUILD_LOAD"

cd ..

# ====================================================

if [[ ! -d "$SCRIPT_DIR/include/gtsam_factors" ]]; then
    mkdir -p "$SCRIPT_DIR/include/gtsam_factors"
fi

# copy header files to include folder
cp *.h "$SCRIPT_DIR/include/gtsam_factors"
