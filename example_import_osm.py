from zombies import import_osm_roads

try:
    roads = import_osm_roads(place="Piedmont, California, USA")
    print(f"Loaded {len(roads)} roads")
    for road in roads[:5]:
        print(road)
except Exception as e:
    print(f"Failed to import OSM data: {e}")
