import carla
import math
from agents.navigation.basic_agent import BasicAgent
from agents.tools.misc import draw_waypoints

# TODO: This should be configurable
default_opt_dict = {
    "ignore_traffic_light": False,
    "ignore_vehicles": False,
    "sampling_resolution": 2.0,
    "base_vehicle_threshold": 5.0,
    "base_tlight_threshold": 5.0,
    "max_brake": 0.5
}


def distance(point1, point2):
    """Calculate the distance between two points."""
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


class PathTracker:
    '''Path tracker for the agent to follow the path
    '''

    def __init__(self, origin, destination, actor):
        """
        Args:
            origin (Tuple(int, int, int)): The origin of the path.
            destination (Tuple(int, int, int)): The destination of the path.
            actor (carla.Actor): The actor to be controlled.
        """
        self.agent = BasicAgent(actor, opt_dict=default_opt_dict)
        self.agent.set_destination(
            carla.Location(x=destination[0],
                           y=destination[1], z=destination[2]),
            carla.Location(x=origin[0], y=origin[1], z=origin[2])
        )
        self.dest_waypoint = None
        self.last_location = None
        self.distance_cache = 0.0

    def get_orientation_difference_to_end_in_radians(self):
        self.get_path()

        return math.radians(
            math.fabs(self.agent._vehicle.get_transform().rotation.yaw -
                      self.dest_waypoint.transform.rotation.yaw))

    def get_euclidean_distance_to_end(self):
        """Get the euclidean distance to the end of the planned path."""
        path = self.get_path()

        if len(path) > 0:
            node_coords = (path[-1][0].transform.location.x,
                           path[-1][0].transform.location.y)
            actor_coords = self.agent._vehicle.get_location()
            actor_coords = (actor_coords.x, actor_coords.y)

            return distance(node_coords, actor_coords)

        return 0.0

    def get_distance_to_end(self):
        """Get the distance to the end of the planned path."""

        # use cache to accelerate the calculation
        last_loc = self.agent._vehicle.get_location()
        if self.last_location is None or \
                self.last_location.distance(last_loc) >= 0.5:
            self.last_location = last_loc
        else:
            return self.distance_cache

        path = self.get_path()

        if len(path) > 0:
            # actor distance to the nearest node
            node_coords = (path[0][0].transform.location.x,
                           path[0][0].transform.location.y)
            actor_coords = self.agent._vehicle.get_location()
            actor_coords = (actor_coords.x, actor_coords.y)
            dist = distance(node_coords, actor_coords)

            # iterate over the path and add the distance between nodes
            for i in range(1, len(path)):
                node_coords1 = (path[i][0].transform.location.x,
                                path[i][0].transform.location.y)
                node_coords2 = (path[i - 1][0].transform.location.x,
                                path[i - 1][0].transform.location.y)
                dist += distance(node_coords1, node_coords2)
        else:
            return 0.0

        self.distance_cache = dist
        return dist

    def run_step(self):
        """Run one step of navigation.

        Returns:
            carla.VehicleControl
        """
        behavior = self.agent.run_step()
        return behavior

    def draw(self):
        """Draw the planned path on the simulator."""
        waypoints = [wp[0] for wp in self.get_path()]
        draw_waypoints(self.agent._vehicle.get_world(), waypoints)

    def get_path(self):
        """Get the path planned by the local planner.

        The path is dynamically updated by the local planner after each call to
        run_step(). Thus, this function will update the destination waypoint.

        Returns:
            deque[Tuple(carla.Waypoint, RoadOption)]
        """
        path = self.agent._local_planner.get_plan()
        if len(path) > 0:
            self.dest_waypoint = path[-1][0]

        return path
