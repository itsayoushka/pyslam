#!/usr/bin/env bash
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

#N.B: this script allows to build the C++ core of pySLAM

SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used

ROOT_DIR="$SCRIPT_DIR_"
SCRIPTS_DIR="$ROOT_DIR/scripts"

# ====================================================
# import the bash utils 
. "$ROOT_DIR"/bash_utils.sh 

# ====================================================

STARTING_DIR=`pwd`
cd "$ROOT_DIR"

#set -e

print_blue '================================================'
print_blue "Building pySLAM C++ core"
print_blue '================================================'

export PYSLAM_BUILD_JOBS="${PYSLAM_BUILD_JOBS:-1}"
export PYSLAM_BUILD_LOAD="${PYSLAM_BUILD_LOAD:-1}"
export PYSLAM_CPP_BUILD_JOBS="${PYSLAM_CPP_BUILD_JOBS:-$PYSLAM_BUILD_JOBS}"
export PYSLAM_CPP_BUILD_LOAD="${PYSLAM_CPP_BUILD_LOAD:-$PYSLAM_BUILD_LOAD}"
print_blue "Using safe build parallelism: jobs=$PYSLAM_CPP_BUILD_JOBS load=$PYSLAM_CPP_BUILD_LOAD"

cd "$ROOT_DIR/pyslam/slam/cpp"
./build.sh

cd "$STARTING_DIR"
