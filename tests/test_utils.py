import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from zombies import is_equal, distance, dot_product, generate_grid_roads, import_osm_roads
import networkx as nx

def test_is_equal_true():
    a=np.array([1,2])
    b=np.array([1,2])
    assert is_equal(a,b)

def test_is_equal_false():
    a=np.array([1,2])
    b=np.array([2,1])
    assert not is_equal(a,b)

def test_distance():
    assert distance(0,0,3,4)==5

def test_dot_product():
    assert dot_product(1,2,3,4)==11

def test_generate_grid_roads_count():
    roads=generate_grid_roads(100,100,20)
    assert len(roads)==50

def test_import_osm_roads_from_graph():
    G = nx.MultiDiGraph()
    G.add_node(0, x=0.0, y=0.0)
    G.add_node(1, x=1.0, y=0.0)
    G.add_node(2, x=1.0, y=1.0)
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    roads = import_osm_roads(graph=G)
    assert len(roads) == 2
    assert all(hasattr(r, 'start') and hasattr(r, 'end') for r in roads)
