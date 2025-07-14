# AGENT Instructions

## Setup
Run `./setup.sh` to create a virtual environment and install the dependencies. The script expects MPI compilers (`mpicc` and `mpic++`) to be available so that `repast4py` can be built. It installs a CPU-only build of PyTorch first to avoid downloading large GPU binaries.

## Testing
Execute `pytest` from the repository root. The tests rely on the packages listed in `requirements.txt`.

## Style
Follow [PEP8](https://peps.python.org/pep-0008/) conventions and format Python code with `black`.

