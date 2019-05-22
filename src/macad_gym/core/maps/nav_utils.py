import math
import numpy as np
import sys

import carla

sys.path.append("macad_gym/carla/PythonAPI/")
from macad_gym.carla.PythonAPI.agents.navigation.local_planner import \
    RoadOption  # noqa: E402
from macad_gym.carla.PythonAPI.agents.tools.misc import vector  # noqa: E402


def get_shortest_path_distance(world, planner, origin, destination):
    """
    This function calculates the distance of the shortest path connecting
    origin and destination using A* search with distance heuristic.
    Args:
        world: carla world object
        planner: carla.macad_agents.navigation's Global route planner object
        origin (tuple): Origin (x, y, z) position on the map
        destination (tuple): Destination (x, y, z) position on the map

    Returns:
        The shortest distance from origin to destination along a feasible path

    """
    waypoints = get_shortest_path_waypoints(world, planner, origin,
                                            destination)
    distance = 0.0
    for i in range(1, len(waypoints)):
        l1 = waypoints[i - 1][0].transform.location
        l2 = waypoints[i][0].transform.location

        distance += math.sqrt((l1.x - l2.x) * (l1.x - l2.x) + (l1.y - l2.y) *
                              (l1.y - l2.y) + (l1.z - l2.z) * (l1.z - l2.z))
    return distance


def get_shortest_path_waypoints(world, planner, origin, destination):
    """
    Return a list of waypoints along a shortest-path.
    Adapted from BasicAgent.set_destination.

    Uses A* planner to find the shortest path and returns a list of waypoints.
    Useful for trajectory planning and control or for drawing the waypoints.

    Args:
        world: carla world object
        planner: carla.macad_agents.navigation's Global route planner object
        origin (tuple): Origin (x, y, z) position on the map
        destination (tuple): Destination (x, y, z) position on the map

    Returns:
        A list of waypoints with corresponding actions connecting the origin
        and the destination on the map along the shortest path.

    """

    start_waypoint = world.get_map().get_waypoint(carla.Location(*origin))
    end_waypoint = world.get_map().get_waypoint(carla.Location(*destination))
    solution = []
    hop_resolution = 2.0

    # Setting up global router
    # planner.setup()

    # Obtain route plan
    x1 = start_waypoint.transform.location.x
    y1 = start_waypoint.transform.location.y
    x2 = end_waypoint.transform.location.x
    y2 = end_waypoint.transform.location.y
    route = planner.plan_route((x1, y1), (x2, y2))

    current_waypoint = start_waypoint
    route.append(RoadOption.VOID)
    for action in route:

        #   Generate waypoints to next junction
        wp_choice = current_waypoint.next(hop_resolution)
        while len(wp_choice) == 1:
            current_waypoint = wp_choice[0]
            solution.append((current_waypoint, RoadOption.LANEFOLLOW))
            wp_choice = current_waypoint.next(hop_resolution)
            #   Stop at destination
            if current_waypoint.transform.location.distance(
                    end_waypoint.transform.location) < hop_resolution:
                break
        if action == RoadOption.VOID:
            break

        #   Select appropriate path at the junction
        if len(wp_choice) > 1:

            # Current heading vector
            current_transform = current_waypoint.transform
            current_location = current_transform.location
            projected_location = current_location + carla.Location(
                x=math.cos(math.radians(current_transform.rotation.yaw)),
                y=math.sin(math.radians(current_transform.rotation.yaw)))
            v_current = vector(current_location, projected_location)

            direction = 0
            if action == RoadOption.LEFT:
                direction = 1
            elif action == RoadOption.RIGHT:
                direction = -1
            elif action == RoadOption.STRAIGHT:
                direction = 0
            select_criteria = float('inf')

            #   Choose correct path
            for wp_select in wp_choice:
                v_select = vector(current_location,
                                  wp_select.transform.location)
                cross = float('inf')
                if direction == 0:
                    cross = abs(np.cross(v_current, v_select)[-1])
                else:
                    cross = direction * np.cross(v_current, v_select)[-1]
                if cross < select_criteria:
                    select_criteria = cross
                    current_waypoint = wp_select

            #   Generate all waypoints within the junction
            #   along selected path
            solution.append((current_waypoint, action))
            current_waypoint = current_waypoint.next(hop_resolution)[0]
            while current_waypoint.is_intersection:
                solution.append((current_waypoint, action))
                current_waypoint = current_waypoint.next(hop_resolution)[0]

    return solution


def draw_shortest_path(world, planner, origin, destination):
    """Draws shortest feasible lines/arrows from origin to destination

    Args:
        world:
        planner:
        origin (tuple): (x, y, z)
        destination (tuple): (x, y, z)

    Returns:
        next waypoint as a list of coordinates (x,y,z)
    """
    hops = get_shortest_path_waypoints(world, planner, origin, destination)

    for i in range(1, len(hops)):
        hop1 = hops[i - 1][0].transform.location
        hop2 = hops[i][0].transform.location
        hop1.z = origin[2]
        hop2.z = origin[2]
        if i == len(hops) - 1:
            world.debug.draw_arrow(
                hop1,
                hop2,
                life_time=1.0,
                color=carla.Color(0, 255, 0),
                thickness=0.5)
        else:
            world.debug.draw_line(
                hop1,
                hop2,
                life_time=1.0,
                color=carla.Color(0, 255, 0),
                thickness=0.5)


def get_next_waypoint(world, location, distance=1.0):
    """Return the waypoint coordinates `distance` meters away from `location`

    Args:
        world (carla.World): world to navigate in
        location (tuple): [x, y, z]
        distance (float): Desired separation distance in meters

    Returns:
        The next waypoint as a list of coordinates (x,y,z)
    """
    # TODO: Use named tuple for location
    current_waypoint = world.get_map().get_waypoint(
        carla.Location(location[0], location[1], location[2]))
    current_coords = current_waypoint.transform.location
    next_waypoints = current_waypoint.next(distance)
    if len(next_waypoints) > 0:
        current_coords = next_waypoints[0].transform.location
    return [current_coords.x, current_coords.y, current_coords.z]


def get_shortest_path_distance_old(planner, origin, destination):
    """
    This function calculates the distance of the shortest path connecting
    origin and destination using A* search with distance heuristic.
    Args:
        planner: Global route planner
        origin (tuple): Tuple containing x, y co-ordinates of start position
        destination (tuple): (x, y) co-coordinates of destination position

    Returns:
        The shortest distance from origin to destination along a feasible path

    """
    graph, _ = planner.build_graph()
    path = planner.path_search(origin, destination)
    distance = 0.0
    if len(path) > 0:
        first_node = graph.nodes[path[0]]["vertex"]
        distance = planner.distance(origin, first_node)
        for i in range(1, len(path)):
            distance += planner.distance(graph.nodes[path[i - 1]]["vertex"],
                                         graph.nodes[path[i]]["vertex"])

    return distance


def get_shortest_path_waypoints_old(planner, origin, destination):
    """Return a list of waypoints along a shortest-path

    Uses A* planner to find the shortest path and returns a list of waypoints.
    Useful for trajectory planning and control or for drawing the waypoints.

    Args:
        planner: carla.macad_agents.navigation's Global route planner object
        origin (tuple): Origin (x, y) position on the map
        destination (tuple): Destination (x, y) position on the map

    Returns:
        A list of waypoints connecting the origin and the destination on the map
        along the shortest path.

    """

    graph, xy_id_map = planner.build_graph()
    path = planner.path_search(origin, destination)
    xy_list = []
    for node in path:
        xy_list.append(graph.nodes[node]["vertex"])
    return xy_list


def draw_shortest_path_old(world, planner, origin, destination):
    """Draws shortest feasible lines/arrows from origin to destination

    Args:
        world:
        planner:
        origin (typle): (x, y, z)
        destination:

    Returns:

    """
    xys = get_shortest_path_waypoints(planner, (origin[0], origin[1]),
                                      destination)
    if len(xys) > 2:
        for i in range(len(xys) - 2):
            world.debug.draw_line(
                carla.Location(*xys[i]),
                carla.Location(*xys[i + 1]),
                life_time=1.0,
                color=carla.Color(0, 255, 0))
    elif len(xys) == 2:
        world.debug.draw_arrow(
            carla.Location(*xys[-2]),
            carla.Location(*xys[-1]),
            life_time=100.0,
            color=carla.Color(0, 255, 0),
            thickness=0.5)


class PathTracker(object):
    def __init__(self, world, planner, origin, destination, actor):
        self.world = world
        self.planner = planner
        self.origin = origin
        self.destination = destination
        self.actor = actor
        self.path = []
        self.path_index = 0
        self.generate_path()
        self.last_location = None
        self.distance_cache = 0.0

    def generate_path(self):
        self.last_location = None
        self.set_path(
            get_shortest_path_waypoints(
                self.world, self.planner, self.origin,
                get_next_waypoint(self.world, self.destination)))
        self.path_index = 0

    def advance_path(self):
        if self.path_index < len(self.path):
            last_dist = self.path[self.path_index][0].transform.location.\
                distance(self.actor.get_location())

            for i in range(self.path_index + 1, len(self.path)):
                dist = self.path[i][0].transform.location.\
                    distance(self.actor.get_location())
                if dist <= last_dist:
                    self.path_index = i
                else:
                    break
                last_dist = dist

    def seek_closest(self):
        if len(self.path) > 0:
            close_dist = self.path[0][0].transform.location.\
                distance(self.actor.get_location())
            close_i = 0

            for i in range(1, len(self.path)):
                cur_dist = self.path[i][0].transform.location.\
                    distance(self.actor.get_location())
                if cur_dist <= close_dist:
                    close_i = i

            self.path_index = close_i

    def get_distance_to_end(self):
        last_loc = self.actor.get_location()
        if self.last_location is None or \
                self.last_location.distance(last_loc) >= 0.5:
            self.advance_path()
            self.last_location = last_loc
        else:
            return self.distance_cache

        if self.path_index < len(self.path):
            node_coords = (self.path[self.path_index][0].transform.location.x,
                           self.path[self.path_index][0].transform.location.y)
            actor_coords = self.actor.get_location()
            actor_coords = (actor_coords.x, actor_coords.y)
            distance = self.planner.distance(node_coords, actor_coords)

            for i in range(self.path_index + 1, len(self.path)):
                node_coords1 = (self.path[i - 1][0].transform.location.x,
                                self.path[i - 1][0].transform.location.y)
                node_coords2 = (self.path[i][0].transform.location.x,
                                self.path[i][0].transform.location.y)
                distance += self.planner.distance(node_coords1, node_coords2)
        else:
            return 9999.9

        self.distance_cache = distance
        return distance

    def get_euclidean_distance_to_end(self):
        if len(self.path) > 0:
            node_coords = (self.path[-1][0].transform.location.x,
                           self.path[-1][0].transform.location.y)
            actor_coords = self.actor.get_location()
            actor_coords = (actor_coords.x, actor_coords.y)
            return self.planner.distance(node_coords, actor_coords)
        return 9999.0

    def get_orientation_difference_to_end_in_radians(self):
        if len(self.path) > 0:
            return math.radians(
                math.fabs(self.actor.get_transform().rotation.yaw -
                          self.path[-1][0].transform.rotation.yaw))
        return math.pi

    def draw(self):
        actor_z = self.actor.get_location().z
        for i in range(self.path_index + 1, len(self.path)):
            hop1 = self.path[i - 1][0].transform.location
            hop2 = self.path[i][0].transform.location
            hop1.z = actor_z
            hop2.z = actor_z
            if i == len(self.path) - 1:
                self.world.debug.draw_arrow(
                    hop1,
                    hop2,
                    life_time=0.5,
                    color=carla.Color(0, 255, 0),
                    thickness=0.5)
            else:
                self.world.debug.draw_line(
                    hop1,
                    hop2,
                    life_time=0.5,
                    color=carla.Color(0, 255, 0),
                    thickness=0.5)

    def reset(self):
        self.path_index = 0

    def set_path(self, path):
        self.path = path
        # self.path = []

        # for p in path:
        #    self.path.append(carla.Location(p[0].transform.location.x,
        #                                    p[0].transform.location.y,
        #                                    p[0].transform.location.z))
