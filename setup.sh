#!/usr/bin/env bash
# Setup script for Repast-RoadBasedMovementProject
# Creates a Python virtual environment and installs dependencies

set -euo pipefail

# Default venv directory
VENV_DIR="venv"

python3 -m venv "$VENV_DIR"
. "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Setup complete. Activate the environment with 'source $VENV_DIR/bin/activate'"
