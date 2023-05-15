# noqa
import carla
from carla_gym.core.world_objects.sensors import CollisionSensor, LaneInvasionSensor

DISCRETE_ACTIONS = {
    # coast
    0: [0.0, 0.0],
    # turn left
    1: [0.0, -0.5],
    # turn right
    2: [0.0, 0.5],
    # forward
    3: [1.0, 0.0],
    # brake
    4: [-0.5, 0.0],
    # forward left
    5: [0.5, -0.05],
    # forward right
    6: [0.5, 0.05],
    # brake left
    7: [-0.5, -0.5],
    # brake right
    8: [-0.5, 0.5],
}

GROUND_Z = 22


class VehicleManager:
    """Controller for vehicle objects."""

    def __init__(self, vehicle_config, vehicle_object, traffic_manager, path_tracker):
        """Constructor.

        Args:
            vehicle_config: actor configuration
            vehicle_object: world object
            traffic_manager: world traffic manager instance
            path_tracker: path tracker object attached to the actor
        """
        self._config = vehicle_config
        self._vehicle = vehicle_object
        self._traffic_manager = traffic_manager
        self._path_tracker = path_tracker

        # Spawn collision and lane sensors if necessary
        self._collision_sensor = CollisionSensor(self._vehicle) if vehicle_config.collision_sensor else None
        self._lane_invasion_sensor = LaneInvasionSensor(self._vehicle) if vehicle_config.lane_sensor else None

        self._vehicle.set_autopilot(self._config.auto_control, self._traffic_manager.get_port())

    def read_observation(self):
        """Read observation and return measurement.

        Returns:
            dict: Measurement data.
        """
        if self._config.enable_planner:
            next_command = self._path_tracker.get_path_commands_seq()[-1]
        else:
            next_command = "LANE_FOLLOW"

        collision_vehicles = self._collision_sensor.collision_vehicles
        collision_pedestrians = self._collision_sensor.collision_pedestrians
        collision_other = self._collision_sensor.collision_other
        intersection_otherlane = self._lane_invasion_sensor.offlane
        intersection_offroad = self._lane_invasion_sensor.offroad

        if next_command == "REACH_GOAL":
            distance_to_goal = 0.0
            distance_to_goal_euclidean = 0.0
        elif self._config.enable_planner:
            distance_to_goal = self._path_tracker.get_distance_to_end()
            distance_to_goal_euclidean = self._path_tracker.get_euclidean_distance_to_end()
        else:
            distance_to_goal = -1.0
            distance_to_goal_euclidean = -1.0

        py_measurements = {
            "x": self._vehicle.get_location().x,
            "y": self._vehicle.get_location().y,
            "pitch": self._vehicle.get_transform().rotation.pitch,
            "yaw": self._vehicle.get_transform().rotation.yaw,
            "roll": self._vehicle.get_transform().rotation.roll,
            "forward_speed": self._vehicle.get_velocity().x,
            "distance_to_goal": distance_to_goal,
            "distance_to_goal_euclidean": distance_to_goal_euclidean,
            "collision_vehicles": collision_vehicles,
            "collision_pedestrians": collision_pedestrians,
            "collision_other": collision_other,
            "intersection_offroad": intersection_offroad,
            "intersection_otherlane": intersection_otherlane,
            "next_command": next_command,
        }

        return py_measurements

    def apply_control(self, throttle, steer, brake, hand_brake, reverse):
        """Apply new control commands to the vehicle object.

        Args:
            throttle: throttle value
            steer: steer value
            brake: brake value
            hand_brake: hand_brake bool value
            reverse: reverse bool value

        Returns:
            N/A.
        """
        if self._config.auto_control:
            if getattr(self._vehicle, "set_autopilot", 0):
                self._vehicle.set_autopilot(True, self._traffic_manager.get_port())
        else:
            self._vehicle.apply_control(
                carla.VehicleControl(throttle=throttle, steer=steer, brake=brake, hand_brake=hand_brake, reverse=reverse)
            )

    def __del__(self):
        """Delete instantiated sub-elements."""
        if self._collision_sensor is not None and self._collision_sensor.sensor.is_alive:
            self._collision_sensor.sensor.destroy()
        if self._lane_invasion_sensor is not None and self._lane_invasion_sensor.sensor.is_alive:
            self._lane_invasion_sensor.sensor.destroy()
