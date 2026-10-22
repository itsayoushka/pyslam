#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd ${SCRIPT_DIR}

# ====================================================
# check if we have external options
EXTERNAL_OPTIONS=$@
if [[ -n "$EXTERNAL_OPTIONS" ]]; then
    echo "external option: $EXTERNAL_OPTIONS" 
fi

OpenCV_DIR="$SCRIPT_DIR/../opencv/install/lib/cmake/opencv4"
if [[ -d "$OpenCV_DIR" ]]; then
    EXTERNAL_OPTIONS="$EXTERNAL_OPTIONS -DOpenCV_DIR=$OpenCV_DIR"
fi 

PYTHON_EXE=${Python3_EXECUTABLE:-${PYTHON_EXE:-$(which python3)}}

if [[ "$OSTYPE" == "darwin"* ]]; then
    # Make sure we don't accidentally use a Linux cross-compiler or Linux sysroot from conda
    unset CC CXX CFLAGS CXXFLAGS LDFLAGS CPPFLAGS SDKROOT CONDA_BUILD_SYSROOT CONDA_BUILD_CROSS_COMPILATION

    # Ask Xcode for the proper macOS SDK path (fallback to default if unavailable)
    MAC_SYSROOT=$(xcrun --show-sdk-path 2>/dev/null || echo "")

    MAC_OPTIONS="-DCMAKE_C_COMPILER=/usr/bin/clang \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++"

    echo "Using MAC_OPTIONS for cpp build: $MAC_OPTIONS"
fi

echo "EXTERNAL_OPTIONS: $EXTERNAL_OPTIONS"
echo "Using Python executable: $PYTHON_EXE"

if [[ -z "$PYSLAM_DBOW_BUILD_JOBS" ]]; then
    BUILD_JOBS=2
else
    BUILD_JOBS="$PYSLAM_DBOW_BUILD_JOBS"
fi
if [[ -z "$PYSLAM_DBOW_BUILD_LOAD" ]]; then
    BUILD_LOAD=2
else
    BUILD_LOAD="$PYSLAM_DBOW_BUILD_LOAD"
fi
echo "Using build parallelism: jobs=$BUILD_JOBS load_limit=$BUILD_LOAD"

# ====================================================

cd modules/dbow2
if [ ! -d build ]; then
    mkdir build
fi
cd build 
cmake .. -DCMAKE_BUILD_TYPE=Release $EXTERNAL_OPTIONS $MAC_OPTIONS
make -j"$BUILD_JOBS" -l"$BUILD_LOAD"

cd ${SCRIPT_DIR}
if [ ! -d build ]; then
    mkdir build
fi
cd build 
cmake .. -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=$PYTHON_EXE $EXTERNAL_OPTIONS $MAC_OPTIONS
make -j"$BUILD_JOBS" -l"$BUILD_LOAD"
