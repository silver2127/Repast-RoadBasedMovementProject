#!/usr/bin/env bash
# Setup script for Repast-RoadBasedMovementProject
# Creates a Python virtual environment and installs dependencies

set -euo pipefail

# Default venv directory
VENV_DIR="venv"

python3 -m venv "$VENV_DIR"
. "$VENV_DIR/bin/activate"

# Use MPI compilers when building packages that require MPI (e.g. repast4py)
export MPICC=mpicc
export MPICXX=mpic++
export CC=mpicc
export CXX=mpic++

python -m pip install --upgrade pip
# Install CPU-only PyTorch first so repast4py doesn't pull in GPU builds
python -m pip install "torch==2.2.2+cpu" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt

echo "Setup complete. Activate the environment with 'source $VENV_DIR/bin/activate'"
