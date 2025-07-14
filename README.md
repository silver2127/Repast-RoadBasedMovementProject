# Repast-RoadBasedMovementProject

The Overall project is my attempt at creating agent based movement along a series of roads. Futher functionality including importing roads from real world maps is along the way.
For Configuring the file's parameters edit the YAML with how you see fit, right not the program is still wip and functions are set with constants specifically the visualizer needs to know the world size.

Running the program only works on linux or WSL
after running the program you can run the visualizer on  windows or linux and it will generate a series of pictures showing the x and y of each agent.

## Setup

Use the provided `setup.sh` script to create a virtual environment and install the required Python dependencies:

```bash
./setup.sh
```

After running the script, activate the environment with:

```bash
source venv/bin/activate
```

The dependencies are listed in `requirements.txt`.

## Importing Roads from OpenStreetMap

An optional utility allows creating `Road` objects directly from OpenStreetMap data. The function fetches a street network using [OSMnx](https://github.com/gboeing/osmnx) and converts each edge into a road.

```python
from zombies import import_osm_roads

roads = import_osm_roads(place="Piedmont, California, USA")
print(len(roads))  # number of road segments retrieved
```

Internet access is required for this example because the street network is downloaded from OpenStreetMap. If the network request fails, check your connectivity or proxy settings.

## Performance Benchmarking

A small benchmarking utility `benchmark_osm.py` imports road networks for a few
places and reports how long each download and conversion takes:

```bash
python benchmark_osm.py
```

This can be useful when experimenting with different open source maps to see how
import time scales with network size.
