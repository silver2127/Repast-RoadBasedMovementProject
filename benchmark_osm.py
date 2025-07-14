import time
from typing import List

from zombies import import_osm_roads

PLACES: List[str] = [
    "Piedmont, California, USA",
    "Modena, Italy",
    "Monaco",
]


def benchmark_import() -> None:
    print("Benchmarking OSM import speed...")
    for place in PLACES:
        start = time.perf_counter()
        try:
            roads = import_osm_roads(place=place)
            duration = time.perf_counter() - start
            print(f"{place}: {len(roads)} roads in {duration:.2f}s")
        except Exception as exc:
            duration = time.perf_counter() - start
            print(f"{place}: failed after {duration:.2f}s: {exc}")


if __name__ == "__main__":
    benchmark_import()
