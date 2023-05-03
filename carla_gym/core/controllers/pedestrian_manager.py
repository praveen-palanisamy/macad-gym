# noqa
import math

import carla
from core.constants import DISTANCE_TO_GOAL_THRESHOLD
from core.world_objects.sensors import CollisionSensor


class PedestrianManager:
    """Controller for pedestrian objects."""

    def __init__(self, pedestrian_config, pedestrian_object, planner, destination):
        """Constructor.

        Args:
            pedestrian_config: actor configuration
            pedestrian_object: world object
            planner: global planner instance
            destination: carla.Location object
        """
        self._config = pedestrian_config
        self._pedestrian = pedestrian_object
        self._planner = planner
        self._destination = destination

        # Spawn collision and lane sensors if necessary
        self._collision_sensor = CollisionSensor(self._pedestrian, 0) if pedestrian_config.collision_sensor else None
        if self._config.auto_control:
            self._pedestrian.controller.start()
            self._pedestrian.controller.go_to_location(carla.Location(*self._destination[:3]))
        else:
            self._pedestrian.controller.stop()

    def read_observation(self):
        """Read observation and return measurement.

        Returns:
            dict: measurement data.
        """
        collision_vehicles = self._collision_sensor.collision_vehicles
        collision_pedestrians = self._collision_sensor.collision_pedestrians
        collision_other = self._collision_sensor.collision_other

        distance_to_goal_euclidean = self._planner.distance(
            (self._pedestrian.get_location().x, self._pedestrian.get_location().y), self._destination[:2]
        )

        if distance_to_goal_euclidean < DISTANCE_TO_GOAL_THRESHOLD:
            next_command = "REACH_GOAL"
        else:
            next_command = "LANE_FOLLOW"

        py_measurements = {
            "x": self._pedestrian.get_location().x,
            "y": self._pedestrian.get_location().y,
            "yaw": self._pedestrian.get_transform().rotation.yaw,
            "x_dir": math.cos(math.radians(self._pedestrian.get_transform().rotation.yaw)),
            "y_dir": math.cos(math.radians(self._pedestrian.get_transform().rotation.yaw)),
            "forward_speed": self._pedestrian.get_velocity().x,
            "distance_to_goal": distance_to_goal_euclidean,
            "distance_to_goal_euclidean": distance_to_goal_euclidean,
            "collision_vehicles": collision_vehicles,
            "collision_pedestrians": collision_pedestrians,
            "collision_other": collision_other,
            "intersection_offroad": 0,
            "intersection_otherlane": 0,
            "next_command": next_command,
        }

        return py_measurements

    def apply_control(self, throttle, steer):
        """Apply new control commands to the pedestrian object.

        Args:
            throttle: throttle value
            steer: steer value

        Returns:
            N/A.
        """
        if not self._config.auto_control:
            rotation = self._pedestrian.get_transform().rotation
            rotation.yaw += steer * 10.0
            x_dir = math.cos(math.radians(rotation.yaw))
            y_dir = math.sin(math.radians(rotation.yaw))

            self._pedestrian.apply_control(
                carla.WalkerControl(speed=3.0 * throttle, direction=carla.Vector3D(x_dir, y_dir, 0.0))
            )

    def __del__(self):
        """Delete instantiated sub-elements."""
        if self._collision_sensor is not None and self._collision_sensor.sensor.is_alive:
            self._collision_sensor.sensor.destroy()
