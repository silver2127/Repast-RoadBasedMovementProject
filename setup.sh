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
python -m pip install -r requirements.txt

echo "Setup complete. Activate the environment with 'source $VENV_DIR/bin/activate'"
