import sys
import math
import numpy as np
from typing import Dict, Tuple, List

try:
    from mpi4py import MPI
except Exception:  # pragma: no cover - MPI may be unavailable during testing
    MPI = None
from dataclasses import dataclass
import csv
from scipy.spatial import KDTree
import numba
from numba import int32, int64
from numba.experimental import jitclass
import typing

try:
    from repast4py import core, space, schedule, logging, random
    from repast4py import context as ctx
    from repast4py.parameters import create_args_parser, init_params
    from repast4py.space import ContinuousPoint as cpt
    from repast4py.space import DiscretePoint as dpt
    from repast4py.space import BorderType, OccupancyType
except Exception:  # pragma: no cover - repast4py may be unavailable during testing

    class _DummyAgent:
        pass

    class _DummyCore:
        Agent = object

    core = _DummyCore()
    space = schedule = logging = random = ctx = None
    create_args_parser = init_params = lambda *args, **kwargs: None
    cpt = dpt = None
    BorderType = OccupancyType = None
BATCH_SIZE = 1000
model = None
RADIUSSEARCH = 20
ROADLENGTH = 20


@numba.jit((int64[:], int64[:]), nopython=True)
def is_equal(a1, a2):
    return a1[0] == a2[0] and a1[1] == a2[1]


def write_agent_to_csv(
    tick, agent_type, agent_id, x, y, status=None, filename="agents_status.csv"
):
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        if status is None:
            writer.writerow([tick, agent_type, agent_id, x, y])
        else:
            writer.writerow([tick, agent_type, agent_id, x, y, status])


RADIUS = 5  # define your search radius
RADIUSH = 10  # /???????


@numba.jit(nopython=True)
def get_extended_neighbors(x, y, radius):
    extended_nghs = [
        (x + i, y + j)
        for i in range(-radius, radius + 1)
        for j in range(-radius, radius + 1)
        if not (i == 0 and j == 0)
    ]
    return extended_nghs


@numba.jit(nopython=True)
def distance(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


@numba.jit(nopython=True)
def dot_product(x1, y1, x2, y2):
    return x1 * x2 + y1 * y2


spec = [
    ("mo", int32[:]),
    ("no", int32[:]),
    ("xmin", int32),
    ("ymin", int32),
    ("ymax", int32),
    ("xmax", int32),
]


@jitclass(spec)
class GridNghFinder:

    def __init__(self, xmin, ymin, xmax, ymax):
        self.mo = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=np.int32)
        self.no = np.array([1, 1, 1, 0, 0, 0, -1, -1, -1], dtype=np.int32)
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def find(self, x, y):
        xs = self.mo + x
        ys = self.no + y

        xd = (xs >= self.xmin) & (xs <= self.xmax)
        xs = xs[xd]
        ys = ys[xd]

        yd = (ys >= self.ymin) & (ys <= self.ymax)
        xs = xs[yd]
        ys = ys[yd]

        return np.stack((xs, ys, np.zeros(len(ys), dtype=np.int32)), axis=-1)


@dataclass
class Road:
    """Represents a road segment."""

    start: Tuple[int, int]
    end: Tuple[int, int]
    capacity: int = 12
    utilization: int = 0
    oldtick: int = 0
    closed_ticks_remaining: int = 0

    def close(self, duration: int = 1):
        """Close the road for ``duration`` ticks."""
        self.closed_ticks_remaining = max(duration, 1)

    def open(self):
        """Open the road immediately."""
        self.closed_ticks_remaining = 0

    def refreshroad(self, tick):
        if self.utilization == self.capacity and self.oldtick != tick:
            self.utilization = self.utilization - self.capacity
            print("Refreshed successfully")

        return

    def traverse_road(self, agent: "Human", model, step_size=10):
        if self.closed_ticks_remaining > 0:
            return
        if self.utilization == self.capacity:
            self.refreshroad(model.runner.schedule.tick)
            return
        current_position = model.space.get_location(agent)
        dir_x = (self.end[0] - current_position.x) / distance(
            current_position.x, current_position.y, self.end[0], self.end[1]
        )
        dir_y = (self.end[1] - current_position.y) / distance(
            current_position.x, current_position.y, self.end[0], self.end[1]
        )
        new_x = current_position.x + dir_x * step_size
        new_y = current_position.y + dir_y * step_size
        model.move(agent, new_x, new_y)
        self.utilization += 1
        self.refreshroad(model.runner.schedule.tick)
        self.oldtick = model.runner.schedule.tick

    def traverse_backwards(self, agent: "Human", model, step_size=10):
        if self.closed_ticks_remaining > 0:
            return
        if self.utilization == self.capacity:
            return
        current_position = model.space.get_location(agent)
        dir_x = (self.start[0] - current_position.x) / distance(
            current_position.x, current_position.y, self.start[0], self.start[1]
        )
        dir_y = (self.start[1] - current_position.y) / distance(
            current_position.x, current_position.y, self.start[0], self.start[1]
        )
        new_x = current_position.x + dir_x * step_size
        new_y = current_position.y + dir_y * step_size
        model.move(agent, new_x, new_y)
        self.utilization += 1
        self.refreshroad(model.runner.schedule.tick)
        self.oldtick = model.runner.schedule.tick


def generate_grid_roads(max_width: int, max_height: int, road_length: int = ROADLENGTH):
    roads = []

    # Vertical roads
    for x in range(0, max_width, road_length):
        start_y = 0
        end_y = start_y + road_length

        while end_y <= max_height:
            roads.append(Road((x, start_y), (x, end_y)))
            start_y = end_y
            end_y = start_y + road_length

        # Horizontal roads
    for y in range(0, max_height, road_length):
        start_x = 0
        end_x = start_x + road_length

        while end_x <= max_width:
            roads.append(Road((start_x, y), (end_x, y)))
            start_x = end_x
            end_x = start_x + road_length

    return roads


def import_osm_roads(
    place: str = None, network_type: str = "drive", graph=None
) -> List[Road]:
    """Import road data from OpenStreetMap and return it as a list of ``Road`` objects.

    Parameters
    ----------
    place : str, optional
        A place name understood by OpenStreetMap (e.g. ``"Berkeley, California"``).
        Ignored if ``graph`` is provided.
    network_type : str, default "drive"
        The type of street network to retrieve when using ``place``.
    graph : networkx.MultiDiGraph, optional
        A pre-built OSMnx graph. If given, this graph is used instead of
        downloading data. This is useful for testing.

    Returns
    -------
    List[Road]
        A list of ``Road`` instances created from the OSM data.
    """
    try:
        import osmnx as ox  # type: ignore
    except ImportError as exc:
        raise ImportError("osmnx is required to import OSM roads") from exc

    if graph is None:
        if place is None:
            raise ValueError("Either 'place' or 'graph' must be provided")
        graph = ox.graph_from_place(place, network_type=network_type)

    roads: List[Road] = []
    for u, v, _ in graph.edges(data=True):
        start_node = graph.nodes[u]
        end_node = graph.nodes[v]
        start = (int(start_node["x"]), int(start_node["y"]))
        end = (int(end_node["x"]), int(end_node["y"]))
        roads.append(Road(start, end))

    return roads


class Human(core.Agent):
    """The Human Agent

    Args:
        a_id: a integer that uniquely identifies this Human on its starting rank
        rank: the starting MPI rank of this Human.
    """

    TYPE = 0

    def __init__(self, a_id: int, rank: int):
        super().__init__(id=a_id, type=Human.TYPE, rank=rank)
        self.infected = False
        self.infected_duration = 0
        self.recent_roads = []

    def save(self) -> Tuple:
        """Saves the state of this Human as a Tuple.

        Used to move this Human from one MPI rank to another.

        Returns:
            The saved state of this Human.
        """
        return (self.uid, self.infected, self.infected_duration)

    def infect(self):
        self.infected = True

    def choose_best_road_towards_target(self, target_x=500, target_y=500):
        # Get the agent's current position
        """
        current_position = model.space.get_location(self)

        # Find all nearby roads
        nearby_roads = model.find_nearby_roads(current_position.x, current_position.y)

        best_road = None
        best_distance = float('inf')
        for road in nearby_roads:
            # Check if road leads closer to the target
            current_dist = distance(current_position.x, current_position.y, target_x, target_y)
            end_dist = distance(road.end[0], road.end[1], target_x, target_y)
            start_dist = distance(road.start[0], road.start[1], target_x, target_y)

            # If moving along this road gets us closer to the target
            if end_dist < current_dist or start_dist < current_dist:
                # Check if it's the best option so far
                road_dist = min(end_dist, start_dist)
                if road_dist < best_distance:
                    best_distance = road_dist
                    best_road = road

        return best_road
        """
        current_position = model.space.get_location(self)
        return model.find_nth_best_road(current_position.x, current_position.y, 1)

    """
    def step(self):
        cpt = model.space.get_location(self)
        
        model.log_agent_data(model.runner.schedule.tick, "Human", self.uid[0], cpt.x, cpt.y)
            # Sample target point
        x0, y0 = 500, 500
        
        # Get the agent's current position
        current_position = model.space.get_location(self)

        if current_position.x == x0 and current_position.y == y0:
            return

        # Find the best road towards the target
        best_road = self.choose_best_road_towards_target()

        # If no suitable road is found, return
        if best_road is None:
            return
        flag = False
        # Check if the best road is in the agent's recent memory
        if best_road in self.recent_roads:
            index = self.recent_roads.index(best_road)
            best_road = model.find_nth_best_road(current_position.x, current_position.y, 1)#model.find_nth_best_road(current_position.x, current_position.y, index + 1)
            if best_road is None:  # If there's no alternative, then return
                flag = True
                return

        # Update the agent's memory
        self.recent_roads.append(best_road)
        if len(self.recent_roads) > 3:  # Remember only the last 3 roads
            self.recent_roads.pop(0)
        
        if flag:
            self.recent_roads[3].traverse_road(self,model)
        # If the agent's current position is closer to the road's end, traverse backwards
        if distance(current_position.x, current_position.y, best_road.end[0], best_road.end[1]) < \
        distance(current_position.x, current_position.y, best_road.start[0], best_road.start[1]):
            best_road.traverse_backwards(self,model)
        else:
            best_road.traverse_road(self,model)

        

        return  
        """

    def step(self):
        cpt = model.space.get_location(self)

        model.log_agent_data(
            model.runner.schedule.tick, "Human", self.uid[0], cpt.x, cpt.y
        )
        # Sample target point
        x0, y0 = 500, 500

        # Get the agent's current position
        current_position = model.space.get_location(self)

        if current_position.x == x0 and current_position.y == y0:
            return

        # Find the best road towards the target
        best_road = self.choose_best_road_towards_target()
        # If no suitable road is found, return
        if best_road == None:
            return
        if distance(
            current_position.x, current_position.y, best_road.end[0], best_road.end[1]
        ) < distance(
            current_position.x,
            current_position.y,
            best_road.start[0],
            best_road.start[1],
        ):
            best_road.traverse_backwards(self, model)
        else:
            best_road.traverse_road(self, model)

    # cooler roads pathfinding closed roads, prefered roads etc..
    # visualizing roads.
    # less roads
    # agents are cars
    # Certain agents for certain roads?
    # roads can get to capacity
    # Routes to take that are prefered "Via alert? "
    # Too hard to turn around? road clogging from backing
    # Accidents
    # Things getting clog


class Zombie(core.Agent):

    TYPE = 1

    def __init__(self, a_id, rank):
        super().__init__(id=a_id, type=Zombie.TYPE, rank=rank)

    def save(self):
        return (self.uid,)

    def step(self):
        grid = model.grid
        pt = grid.get_location(self)
        # nghs = model.ngh_finder.find(pt.x, pt.y)  # include_origin=True)
        nghs = get_extended_neighbors(pt.x, pt.y, RADIUS)
        at = dpt(0, 0)
        maximum = [[], -(sys.maxsize - 1)]
        for ngh in nghs:
            at._reset_from_array(np.array(ngh))
            count = 0
            for obj in grid.get_agents(at):
                if obj.uid[1] == Human.TYPE:
                    count += 1
            if count > maximum[1]:
                maximum[0] = [ngh]
                maximum[1] = count
            elif count == maximum[1]:
                maximum[0].append(ngh)

        max_ngh = maximum[0][random.default_rng.integers(0, len(maximum[0]))]

        if not is_equal(np.array(max_ngh), pt.coordinates):
            # direction = (max_ngh - pt.coordinates[0:3]) * 0.25
            direction = (max_ngh - pt.coordinates[0:2]) * 0.25

            cpt = model.space.get_location(self)
            # timer.start_timer('zombie_move')
            model.move(self, cpt.x + direction[0], cpt.y + direction[1])
            # timer.stop_timer('zombie_move')
        cpt = model.space.get_location(self)
        # write_agent_to_csv(model.runner.schedule.tick, "Zombie", self.uid[0], cpt.x, cpt.y)
        pt = grid.get_location(self)
        for obj in grid.get_agents(pt):
            if obj.uid[1] == Human.TYPE:
                obj.infect()
                break


agent_cache = {}


def restore_agent(agent_data: Tuple):
    """Creates an agent from the specified agent_data.

    This is used to re-create agents when they have moved from one MPI rank to another.
    The tuple returned by the agent's save() method is moved between ranks, and restore_agent
    is called for each tuple in order to create the agent on that rank. Here we also use
    a cache to cache any agents already created on this rank, and only update their state
    rather than creating from scratch.

    Args:
        agent_data: the data to create the agent from. This is the tuple returned from the agent's save() method
                    where the first element is the agent id tuple, and any remaining arguments encapsulate
                    agent state.
    """
    uid = agent_data[0]
    # 0 is id, 1 is type, 2 is rank
    if uid[1] == Human.TYPE:
        if uid in agent_cache:
            h = agent_cache[uid]
        else:
            h = Human(uid[0], uid[2])
            agent_cache[uid] = h

        # restore the agent state from the agent_data tuple
        h.infected = agent_data[1]
        h.infected_duration = agent_data[2]
        return h
    else:
        # note that the zombie has no internal state
        # so there's nothing to restore other than
        # the Zombie itself
        if uid in agent_cache:
            return agent_cache[uid]
        else:
            z = Zombie(uid[0], uid[2])
            agent_cache[uid] = z
            return z


@dataclass
class Counts:
    """Dataclass used by repast4py aggregate logging to record
    the number of Humans and Zombies after each tick.
    """

    humans: int = 0
    zombies: int = 0


class Model:

    def __init__(self, comm, params):
        self.comm = comm
        self.context = ctx.SharedContext(comm)
        self.rank = self.comm.Get_rank()
        self.agent_logs = []
        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_stop(params["stop.at"])
        self.runner.schedule_end_event(self.at_end)

        box = space.BoundingBox(
            0, params["world.width"], 0, params["world.height"], 0, 0
        )
        self.grid = space.SharedGrid(
            "grid",
            bounds=box,
            borders=BorderType.Sticky,
            occupancy=OccupancyType.Multiple,
            buffer_size=2,
            comm=comm,
        )
        self.context.add_projection(self.grid)
        self.space = space.SharedCSpace(
            "space",
            bounds=box,
            borders=BorderType.Sticky,
            occupancy=OccupancyType.Multiple,
            buffer_size=2,
            comm=comm,
            tree_threshold=100,
        )
        self.context.add_projection(self.space)
        self.ngh_finder = GridNghFinder(0, 0, box.xextent, box.yextent)

        self.counts = Counts()
        loggers = logging.create_loggers(self.counts, op=MPI.SUM, rank=self.rank)
        self.data_set = logging.ReducingDataSet(
            loggers, self.comm, params["counts_file"]
        )

        world_size = comm.Get_size()

        total_human_count = params["human.count"]
        pp_human_count = int(total_human_count / world_size)
        if self.rank < total_human_count % world_size:
            pp_human_count += 1

        local_bounds = self.space.get_local_bounds()
        for i in range(pp_human_count):
            h = Human(i, self.rank)
            self.context.add(h)
            x = random.default_rng.uniform(
                local_bounds.xmin, local_bounds.xmin + local_bounds.xextent
            )
            y = random.default_rng.uniform(
                local_bounds.ymin, local_bounds.ymin + local_bounds.yextent
            )
            self.move(h, x, y)

        total_zombie_count = params["zombie.count"]
        pp_zombie_count = int(total_zombie_count / world_size)
        if self.rank < total_zombie_count % world_size:
            pp_zombie_count += 1

        for i in range(pp_zombie_count):
            zo = Zombie(i, self.rank)
            self.context.add(zo)
            x = random.default_rng.uniform(
                local_bounds.xmin, local_bounds.xmin + local_bounds.xextent
            )
            y = random.default_rng.uniform(
                local_bounds.ymin, local_bounds.ymin + local_bounds.yextent
            )
            self.move(zo, x, y)

        self.zombie_id = pp_zombie_count
        with open("agents_status.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Tick", "Agent Type", "Agent ID", "X", "Y", "Status"])
        # Generate the roads
        self.roads = generate_grid_roads(params["world.width"], params["world.height"])

        # Initialize road points and mapping for KD-Tree
        self.road_points = []
        self.road_mapping = {}
        for road in self.roads:
            point = road.end  # or road.start based on your logic
            self.road_points.append(point)
            self.road_mapping[point] = road

        # Create the KD-Tree
        self.tree = KDTree(self.road_points)
        self.kdtree = KDTree(self.road_points)

    def get_all_roads(self):

        return self.roads

    def find_nearby_roads(self, x, y, radius=RADIUSSEARCH):
        """Find all roads within a given radius of a point (x, y) using KDTree."""

        nearby_road_indices = self.tree.query_ball_point([x, y], radius)

        return [self.roads[index] for index in nearby_road_indices]

    # import numpy as np

    def find_nth_best_road(self, x, y, n, destination_x=500, destination_y=500):
        if n <= 0:
            raise ValueError("n should be a positive integer")

        # Calculate the Euclidean distance from the midpoint of each road to the destination
        road_distances = []
        for road in self.roads:
            distance_start = np.sqrt(
                (destination_x - road.start[0]) ** 2
                + (destination_y - road.start[1]) ** 2
            )
            distance_end = np.sqrt(
                (destination_x - road.end[0]) ** 2 + (destination_y - road.end[1]) ** 2
            )
            distance = min(distance_start, distance_end)
            # distance = np.sqrt((destination_x - midpoint_x)**2 + (destination_y - midpoint_y)**2)
            road_distances.append(distance)

        # Sort the road distances and indices in ascending order (shortest distances first)
        sorted_indices = np.argsort(road_distances)

        # Check if n is within the bounds of the list
        if n > len(sorted_indices):
            raise ValueError(
                f"Cannot find the {n}th best road as there are only {len(sorted_indices)} roads."
            )

        # Return the road with the nth-shortest distance (1-indexed)
        return self.roads[sorted_indices[n - 1]]

    def log_agent_data(self, tick, agent_type, agent_id, x, y):
        # Append to the agent_logs but don't write to CSV here
        print(f"{tick} {x} {y}")
        self.agent_logs.append([tick, agent_type, agent_id, x, y])

    # def find_nearest_road(self, x, y):
    # Get the index of the nearest road point
    # nearest_point_index = self.kdtree.query([x, y])[1]
    # Return the road associated with the nearest point
    # return self.road_mapping[self.road_points[nearest_point_index]]

    # def find_best_road_towards_destination(self, x, y, destination_x=500, destination_y=500):
    # Direction vector towards the destination
    #  dir_x = destination_x - x
    # dir_y = destination_y - y

    # Calculate the dot product of the road direction and the direction towards the destination
    #  road_scores = []
    # for road in self.roads:
    #    road_dir_x = road.end[0] - road.start[0]
    #     road_dir_y = road.end[1] - road.start[1]
    #     score = dot_product(dir_x, dir_y, road_dir_x, road_dir_y)
    #    road_scores.append(score)

    # Choose the road with the highest score
    # best_road_index = np.argmax(road_scores)
    # return self.roads[best_road_index]

    def choose_best_road_towards_target(self, x, y, target_x=500, target_y=500):
        # Get the agent's current position
        current_position = model.space.get_location(self)

        # Find all nearby roads
        nearby_roads = model.find_nearby_roads(
            current_position.x, current_position.y
        )  # Implement a function that returns multiple nearby roads, not just the closest

        best_road = None
        best_distance = float("inf")
        for road in nearby_roads:
            # Check if road leads closer to the target
            current_dist = distance(
                current_position.x, current_position.y, target_x, target_y
            )
            end_dist = distance(road.end[0], road.end[1], target_x, target_y)
            start_dist = distance(road.start[0], road.start[1], target_x, target_y)

            # If moving along this road gets us closer to the target
            if end_dist < current_dist or start_dist < current_dist:
                # Check if it's the best option so far
                road_dist = min(end_dist, start_dist)
                if road_dist < best_distance:
                    best_distance = road_dist
                    best_road = road

        return best_road

    def at_end(self):
        self.data_set.close()
        self.write_agent_data_to_csv()

    def move(self, agent, x, y):
        # timer.start_timer('space_move')
        self.space.move(agent, cpt(x, y))
        # timer.stop_timer('space_move')
        # timer.start_timer('grid_move')
        self.grid.move(agent, dpt(int(math.floor(x)), int(math.floor(y))))
        # timer.stop_timer('grid_move')

    def step(self):
        print(str(model.runner.schedule.tick))
        # print("{}: {}".format(self.rank, len(self.context.local_agents)))
        tick = self.runner.schedule.tick
        # update dynamic road conditions
        for road in self.roads:
            if road.closed_ticks_remaining > 0:
                road.closed_ticks_remaining -= 1

        # randomly close a road for a few ticks
        if random.default_rng.random() < 0.05:
            road = random.default_rng.choice(self.roads)
            road.close(duration=3)
        self.log_counts(tick)
        self.context.synchronize(restore_agent)

        # timer.start_timer('z_step')
        for z in self.context.agents(Zombie.TYPE):
            z.step()
        for h in self.context.agents(Human.TYPE):
            h.step()

        # timer.stop_timer('z_step')

        # timer.start_timer('h_step')
        dead_humans = []

    # for h in self.context.agents(Human.TYPE):
    #    dead, pt = h.step()
    #   if dead:
    #      dead_humans.append((h, pt))

    # for h, pt in dead_humans:
    #   model.remove_agent(h)
    #  model.add_zombie(pt)

    # timer.stop_timer('h_step')

    def run(self):
        self.runner.execute()

    def remove_agent(self, agent):
        self.context.remove(agent)

    def add_zombie(self, pt):
        z = Zombie(self.zombie_id, self.rank)
        self.zombie_id += 1
        self.context.add(z)
        self.move(z, pt.x, pt.y)
        # print("Adding zombie at {}".format(pt))

    def log_agent_data(self, tick, agent_type, agent_id, x, y):
        self.agent_logs.append([tick, agent_type, agent_id, x, y])

    def write_agent_data_to_csv(self, filename="agent_data.csv"):
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            # Write the header
            writer.writerow(["Tick", "Agent Type", "Agent ID", "X", "Y"])
            print(self.agent_logs)
            # Write the logs
            writer.writerows(self.agent_logs)
        return

    def log_counts(self, tick):
        # Get the current number of zombies and humans and log
        num_agents = self.context.size([Human.TYPE, Zombie.TYPE])
        self.counts.humans = num_agents[Human.TYPE]
        self.counts.zombies = num_agents[Zombie.TYPE]
        self.data_set.log(tick)

        # Do the cross-rank reduction manually and print the result
        if tick % 10 == 0:
            human_count = np.zeros(1, dtype="int64")
            zombie_count = np.zeros(1, dtype="int64")
            self.comm.Reduce(
                np.array([self.counts.humans], dtype="int64"),
                human_count,
                op=MPI.SUM,
                root=0,
            )
            self.comm.Reduce(
                np.array([self.counts.zombies], dtype="int64"),
                zombie_count,
                op=MPI.SUM,
                root=0,
            )
            if self.rank == 0:
                print(
                    "Tick: {}, Human Count: {}, Zombie Count: {}".format(
                        tick, human_count[0], zombie_count[0]
                    ),
                    flush=True,
                )


def run(params: Dict):
    """Creates and runs the Zombies Model.

    Args:
        params: the model input parameters
    """
    global model
    model = Model(MPI.COMM_WORLD, params)
    model.run()


if __name__ == "__main__":
    parser = create_args_parser()
    args = parser.parse_args()
    params = init_params(args.parameters_file, args.parameters)
    run(params)
