"""Utilities to help the navigation process."""

import math
import sys

import carla
import numpy as np
from carla_gym.carla_api.PythonAPI.agents.navigation.local_planner import RoadOption
from carla_gym.carla_api.PythonAPI.agents.tools.misc import vector
from core.constants import DISTANCE_TO_GOAL_THRESHOLD, ORIENTATION_TO_GOAL_THRESHOLD, ROAD_OPTION_TO_COMMANDS_MAPPING

sys.path.append("macad_gym/carla/PythonAPI/")


def get_shortest_path_distance(world, planner, origin, destination):
    """Compute the distance of the shortest path connecting origin and destination.

    It uses A* search algorithm with a distance heuristic.

    Args:
        world (carla.World): Carla world object.
        planner (carla.macad_agents.navigation.GlobalRoutePlanner): Global route planner object.
        origin (tuple): Origin (x, y, z) position on the map.
        destination (tuple): Destination (x, y, z) position on the map.

    Returns:
        float: The shortest distance from origin to destination along a feasible path.
    """
    waypoints = get_shortest_path_waypoints(world, planner, origin, destination)
    distance = 0.0
    for i in range(1, len(waypoints)):
        l1 = waypoints[i - 1][0].transform.location
        l2 = waypoints[i][0].transform.location

        distance += math.sqrt(
            (l1.x - l2.x) * (l1.x - l2.x) + (l1.y - l2.y) * (l1.y - l2.y) + (l1.z - l2.z) * (l1.z - l2.z)
        )
    return distance


def get_shortest_path_waypoints(world, planner, origin, destination):
    """Return a list of waypoints along a shortest-path.

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
            if current_waypoint.transform.location.distance(end_waypoint.transform.location) < hop_resolution:
                return solution
            if len(wp_choice) > 1 and action.value == RoadOption.VOID.value:
                action = RoadOption.LANEFOLLOW
                route.append(RoadOption.VOID)

        if action.value == RoadOption.VOID.value:
            print(f"Path not correctly created, from {str(origin)} to {str(destination)}")
            break  # safe break

        #   Select appropriate path at the junction
        if len(wp_choice) > 1:
            # Current heading vector
            current_transform = current_waypoint.transform
            current_location = current_transform.location
            projected_location = current_location + carla.Location(
                x=math.cos(math.radians(current_transform.rotation.yaw)),
                y=math.sin(math.radians(current_transform.rotation.yaw)),
            )
            v_current = vector(current_location, projected_location)

            direction = 0
            if action.value == RoadOption.LEFT.value:
                direction = 1
            elif action.value == RoadOption.RIGHT.value:
                direction = -1
            elif action.value == RoadOption.STRAIGHT.value:
                direction = 0
            select_criteria = float("inf")

            #   Choose correct path
            future_wp = []
            for wp in wp_choice:
                future_wp.append(wp.next(2 * hop_resolution)[0])
            for i, wp_select in enumerate(future_wp):
                v_select = vector(current_location, wp_select.transform.location)
                cross = float("inf")
                if direction == 0:
                    cross = abs(np.cross(v_current, v_select)[-1])
                else:
                    cross = direction * np.cross(v_current, v_select)[-1]
                if cross < select_criteria:
                    select_criteria = cross
                    current_waypoint = wp_choice[i]

            # Generate all waypoints within the junction along selected path
            solution.append((current_waypoint, action))
            current_waypoint = current_waypoint.next(hop_resolution)[0]
            while current_waypoint.is_intersection:
                solution.append((current_waypoint, action))
                if current_waypoint.transform.location.distance(end_waypoint.transform.location) < hop_resolution:
                    return solution
                current_waypoint = current_waypoint.next(hop_resolution)[0]

    return solution


def draw_shortest_path(world, planner, origin, destination):
    """Draws shortest feasible lines/arrows from origin to destination.

    Args:
        world: world object
        planner: global planner instance
        origin (tuple): carla.Location object (x, y, z)
        destination (tuple): carla.Location object (x, y, z)

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
            world.debug.draw_arrow(hop1, hop2, life_time=1.0, color=carla.Color(0, 255, 0), thickness=0.5)
        else:
            world.debug.draw_line(hop1, hop2, life_time=1.0, color=carla.Color(0, 255, 0), thickness=0.5)


def get_next_waypoint(world, location, distance=1.0):
    """Return the waypoint coordinates `distance` meters away from `location`.

    Args:
        world (carla.World): world object
        location (tuple): carla.Location object (x, y, z)
        distance (float): desired separation distance in meters

    Returns:
        The next waypoint as a list of coordinates (x,y,z)
    """
    current_waypoint = world.get_map().get_waypoint(carla.Location(*location))
    current_coords = current_waypoint.transform.location
    next_waypoints = current_waypoint.next(distance)
    if len(next_waypoints) > 0:
        current_coords = next_waypoints[0].transform.location
    return [current_coords.x, current_coords.y, current_coords.z]


def get_shortest_path_distance_old(planner, origin, destination):
    """Compute the distance of the shortest path connecting origin and destination.

    It uses A* search algorithm with a distance heuristic.
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
            distance += planner.distance(graph.nodes[path[i - 1]]["vertex"], graph.nodes[path[i]]["vertex"])

    return distance


def get_shortest_path_waypoints_old(planner, origin, destination):
    """Return a list of waypoints along a shortest-path.

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
    """Draws shortest feasible lines/arrows from origin to destination.

    Args:
        world:
        planner:
        origin (tuple): (x, y, z)
        destination:

    Returns:
        N/A.
    """
    xys = get_shortest_path_waypoints(planner, (origin[0], origin[1]), destination)
    if len(xys) > 2:
        for i in range(len(xys) - 2):
            world.debug.draw_line(
                carla.Location(*xys[i]),
                carla.Location(*xys[i + 1]),
                life_time=1.0,
                color=carla.Color(0, 255, 0),
            )
    elif len(xys) == 2:
        world.debug.draw_arrow(
            carla.Location(*xys[-2]),
            carla.Location(*xys[-1]),
            life_time=100.0,
            color=carla.Color(0, 255, 0),
            thickness=0.5,
        )


class PathTracker:
    """Class to manage navigation using waypoints path."""

    def __init__(self, world, planner, actor_object, origin, destination):
        """Constructor.

        Args:
            world: world object
            planner: global planner
            actor_object: object in the space
            origin: origin location of the path
            destination:  destination location of the path
        """
        self.world = world
        self.planner = planner
        self.origin = origin
        self.destination = destination
        self.actor_object = actor_object
        self.path = []
        self.path_index = 0
        self.generate_path()
        self.last_location = None
        self.distance_cache = 0.0

    def generate_path(self):
        """Generate a waypoint path for the actor object location."""
        self.last_location = None
        self.set_path(
            get_shortest_path_waypoints(
                self.world,
                self.planner,
                self.origin,
                get_next_waypoint(self.world, self.destination),
            )
        )
        self.path_index = 0

    def advance_path_index(self):
        """Update the internal path index following the position of the actor object in the world."""
        if self.path_index < len(self.path):
            for i in range(self.path_index + 1, len(self.path)):
                index_dist = self.actor_object.get_location().distance(self.path[self.path_index][0].transform.location)
                next_index_dist = self.actor_object.get_location().distance(self.path[i][0].transform.location)
                step_dist = self.path[self.path_index][0].transform.location.distance(self.path[i][0].transform.location)
                if step_dist <= index_dist and index_dist > next_index_dist:
                    self.path_index = i
                else:
                    if step_dist >= next_index_dist:
                        self.path_index = i + 1 if i + 1 < len(self.path) else i
                    break

    def seek_next_waypoint(self):
        """Update the internal path index, setting the nearest waypoint in the path."""
        assert len(self.path) > 0, "No waypoints in path list."
        i = 0
        for i in range(1, len(self.path)):
            index_dist = self.actor_object.get_location().distance(self.path[i - 1][0].transform.location)
            step_dist = self.path[i - 1][0].transform.location.distance(self.path[i][0].transform.location)
            if step_dist > index_dist and i + 1 < len(self.path):
                i += 1
                break
        self.path_index = i

    def get_distance_to_end(self):
        """Compute the distance from the actor location to the end of the path as sum of waypoints distances."""
        last_loc = self.actor_object.get_location()
        if self.last_location is None or self.last_location.distance(last_loc) >= 0.5:
            self.advance_path_index()
            self.last_location = last_loc
        else:
            return self.distance_cache

        if self.path_index < len(self.path):
            distance = self.last_location.distance(self.path[self.path_index][0].transform.location)
            for i in range(self.path_index + 1, len(self.path)):
                distance += self.path[i - 1][0].transform.location.distance(self.path[i][0].transform.location)
        else:
            return 9999.9

        self.distance_cache = distance
        return distance

    def get_euclidean_distance_to_end(self):
        """Compute the air distance from the actor location to the end of the path as euclidean norm."""
        actor_coords = self.actor_object.get_location()
        dist = float(np.linalg.norm([actor_coords.x - self.destination[0], actor_coords.y - self.destination[1]]) / 100)
        return dist

    def get_orientation_difference_to_end_in_radians(self):
        """Compute the air distance from the actor location to the end of the path in radians."""
        if len(self.path) > 0:
            current = math.radians(self.actor_object.get_transform().rotation.yaw)
            target = math.radians(self.path[-1][0].transform.rotation.yaw)
            return math.fabs(math.cos(current) * math.sin(target) - math.sin(current) * math.cos(target))
        return math.pi

    def get_path_commands_seq(self, debug=False):
        """Get next command to reach the next waypoint in the path."""
        dist = self.get_distance_to_end()
        orientation_diff = self.get_orientation_difference_to_end_in_radians()
        actor_coords = self.actor_object.get_location()
        commands = self.planner.plan_route((actor_coords.x, actor_coords.y), self.destination[:2])
        commands = [ROAD_OPTION_TO_COMMANDS_MAPPING.get(c, "LANE_FOLLOW") for c in commands]
        if len(commands) == 0:
            if dist <= DISTANCE_TO_GOAL_THRESHOLD and orientation_diff <= ORIENTATION_TO_GOAL_THRESHOLD:
                commands = ["REACH_GOAL"]
            else:
                commands = ["LANE_FOLLOW"]
        if debug:
            self.draw()

        return commands

    def draw(self):
        """Debug method to draw the path in the world."""
        actor_z = self.actor_object.get_location().z
        for i in range(self.path_index + 1, len(self.path)):
            hop1 = self.path[i - 1][0].transform.location
            hop2 = self.path[i][0].transform.location
            hop1.z = actor_z
            hop2.z = actor_z
            if i == len(self.path) - 1:
                self.world.debug.draw_arrow(hop1, hop2, life_time=0.5, color=carla.Color(0, 255, 0), thickness=0.5)
            else:
                self.world.debug.draw_line(hop1, hop2, life_time=0.5, color=carla.Color(0, 255, 0), thickness=0.5)

    def plot(self):
        """Debug method to plot coordinates of the path in a 2D graph."""
        import matplotlib.pyplot as plt

        plt.scatter([i[0].transform.location.x for i in self.path], [i[0].transform.location.y for i in self.path])
        plt.gca().invert_yaxis()
        plt.show()

    def reset(self):
        """Reset path index."""
        self.path_index = 0

    def set_path(self, path):
        """Set a path."""
        self.path = path
