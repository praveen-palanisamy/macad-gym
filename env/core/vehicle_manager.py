import pygame
import random
import logging
import numpy as np
import time
import carla
from env.core.sensors.camera_manager import CameraManager
from env.core.sensors.detect_sensors import LaneInvasionSensor, CollisionSensor
from env.core.controllers.keyboard_control import KeyboardControl
from env.core.sensors.hud import HUD
from env.carla.carla.planner.planner import Planner

# Mapping from string repr to one-hot encoding index to feed to the model
COMMAND_ORDINAL = {
    "REACH_GOAL": 0,
    "GO_STRAIGHT": 1,
    "TURN_RIGHT": 2,
    "TURN_LEFT": 3,
    "LANE_FOLLOW": 4,
}

# Carla planner commands
COMMANDS_ENUM = {
    0.0: "REACH_GOAL",
    5.0: "GO_STRAIGHT",
    4.0: "TURN_RIGHT",
    3.0: "TURN_LEFT",
    2.0: "LANE_FOLLOW",
}

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


class VehicleManager(object):
    def __init__(self):
        # General attributes:
        self._world = None
        self._config = None
        self._planner = None  # needed?

        # Vehicle related attributes:
        self._vehicle = None
        self._camera_manager = None
        self._collision_sensor = None
        self._lane_invasion_sensor = None

        # Planner related
        self._planner = None
        self._start_transform = None
        self._end_transform = None
        self._scenario = None

        self.previous_actions = None
        self.previous_reward = None
        self.last_reward = None
        self._start_coord = None
        self._end_coord = None

        # Others
        self._prev_image = None
        self.previous_actions = []
        self.previous_rewards = []
        self._weather = None

        # try to delete this. only used for control two cars concurrently.
        self._parent_list_id = None

    def set_world(self, world):
        """Set world."""
        self._weather = [
            world.get_weather().cloudyness,
            world.get_weather().precipitation,
            world.get_weather().precipitation_deposits,
            world.get_weather().wind_intensity
        ]
        self._world = world

    def set_config(self, config):
        """Set config"""
        self._config = config

    def _set_planner(self):
        """Set planner from city"""
        city = self._config["server_map"].split("/")[-1]
        self._planner = Planner(city)

    def set_vehicle(self, transform):
        """Spawn vehicle.

        Args:
            transform (carla.Transform): start location and rotation.

        Returns:
            N/A.
        """
        # Initialize blueprints and vehicle properties.
        bps = self._world.get_blueprint_library().filter('vehicle')
        bp = random.choice(bps)
        print('spawning vehicle %r with %d wheels' %
              (bp.id, bp.get_attribute('number_of_wheels')))

        # Spawn vehicle.
        vehicle = self._world.try_spawn_actor(bp, transform)
        print('vehicle at ',
              vehicle.get_location().x,
              vehicle.get_location().y,
              vehicle.get_location().z)
        self._vehicle = vehicle

        # Set sensors to the vehicle
        self._set_sensors()

    def _set_sensors(self):
        """Set sensors as needed from config"""

        config = self._config
        cam_state = config["camera"]
        colli_sensor_state = config["collision_sensor"]
        lane_sensor_state = config["lane_sensor"]

        if cam_state == "on":
            # Initialize to be compatible with cam_manager to set HUD.
            pygame.font.init()  # for HUD
            hud = HUD(config["render_x_res"], config["render_y_res"])
            camera_manager = CameraManager(self._vehicle, hud)
            if config["log_images"]:
                # 1: default save method
                # 2: save to memory first
                # We may finally chose one of the two,
                # the two are under test now.
                camera_manager.set_recording_option(1)
            camera_manager.set_sensor(0, notify=False)
            time.sleep(3)  # Wait for the camera to initialize
        self._camera_manager = camera_manager

        if colli_sensor_state == "on":
            self._collision_sensor = CollisionSensor(self._vehicle, 0)
        if lane_sensor_state == "on":
            self._lane_invasion_sensor = LaneInvasionSensor(self._vehicle, 0)

    def set_transform(self, trans):
        """Set transform"""
        self._end_transform = trans

    def set_scenario(self, scenario):
        """Set scenario"""
        self._scenario = scenario

    def apply_control(self, ctrl_args):
        """Apply control to current vehicle.

        Args:
            ctrl_args(list): send control infor, e.g., throttle.

        Returns:
            N/A.

        """
        config = self._config
        if config['manual_control']:
            clock = pygame.time.Clock()
            # pygame
            self._display = pygame.display.set_mode(
                (800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
            logging.debug('pygame started')
            controller = KeyboardControl(self._world, False)
            controller.actor_id = ctrl_args[5]  # only in manual_control mode
            controller.parse_events(self, clock)
            self._on_render()
        elif config["auto_control"]:
            self._vehicle.set_autopilot()
        else:
            # TODO: Planner based on waypoints.
            # cur_location = self._vehicle.get_location()
            # dst_location = carla.Location(
            #   x = self.end_pos[i][0],
            #   y = self.end_pos[i][1],
            #   z = self.end_pos[i][2])
            # cur_map = self.world.get_map()
            # next_point_transform =
            #   get_transform_from_nearest_way_point(
            #       cur_map, cur_location, dst_location)
            # next_point_transform.location.z = 40
            # self.actor_list[i].set_transform(next_point_transform)
            self._vehicle.apply_control(
                carla.VehicleControl(
                    throttle=ctrl_args[0],
                    steer=ctrl_args[1],
                    brake=ctrl_args[2],
                    hand_brake=ctrl_args[3],
                    reverse=ctrl_args[4]))

    # TODO: use the render in cam_manager instead of this.
    def _on_render(self):
        """Render the pygame window."""
        surface = self._camera_manager._surface
        if surface is not None:
            self._display.blit(surface, (0, 0))
        pygame.display.flip()

    def read_observation(self):
        """Read observation and return measurement.

        Returns:
            dict: measurement data.

        """
        cur = self._vehicle
        planner_enabled = self._config["enable_planner"]
        self._set_planner()
        planner = self._planner
        end_loc = self._end_transform.location
        end_rot = self._end_transform.rotation

        if planner_enabled:
            next_command = COMMANDS_ENUM[planner.get_next_command(
                [cur.get_location().x,
                 cur.get_location().y, GROUND_Z], [
                     cur.get_transform().rotation.pitch,
                     cur.get_transform().rotation.yaw, GROUND_Z
                 ], [end_loc.x, end_loc.y, GROUND_Z],
                [end_rot.pitch, end_rot.yaw, GROUND_Z])]
        else:
            next_command = "LANE_FOLLOW"

        collision_vehicles = self._collision_sensor.collision_vehicles
        collision_pedestrians = self._collision_sensor.collision_pedestrians
        collision_other = self._collision_sensor.collision_other
        intersection_otherlane = self._lane_invasion_sensor.offlane
        intersection_offroad = self._lane_invasion_sensor.offroad

        self.previous_actions.append(next_command)
        self.previous_rewards.append(self.last_reward)

        if next_command == "REACH_GOAL":
            distance_to_goal = 0.0  # avoids crash in planner
        elif planner_enabled:
            distance_to_goal = planner.get_shortest_path_distance(
                [cur.get_location().x,
                 cur.get_location().y, GROUND_Z], [
                     cur.get_transform().rotation.pitch,
                     cur.get_transform().rotation.yaw, GROUND_Z
                 ], [end_loc.x, end_loc.y, GROUND_Z],
                [end_rot.pitch, end_rot.yaw, 0]) / 100
        else:
            distance_to_goal = -1

        distance_to_goal_euclidean = float(
            np.linalg.norm([
                cur.get_location().x - end_loc.x,
                cur.get_location().y - end_loc.y
            ]) / 100)

        py_measurements = {
            "x": cur.get_location().x,
            "y": cur.get_location().y,
            "pitch": cur.get_transform().rotation.pitch,
            "yaw": cur.get_transform().rotation.yaw,
            "roll": cur.get_transform().rotation.roll,
            "forward_speed": cur.get_velocity().x,
            "distance_to_goal": distance_to_goal,
            "distance_to_goal_euclidean": distance_to_goal_euclidean,
            "collision_vehicles": collision_vehicles,
            "collision_pedestrians": collision_pedestrians,
            "collision_other": collision_other,
            "intersection_offroad": intersection_offroad,
            "intersection_otherlane": intersection_otherlane,
            "map": self._config["server_map"],
            "current_scenario": self._scenario,
            "next_command": next_command,
            "previous_actions": self.previous_actions,
            "previous_rewards": self.previous_rewards
        }

        return py_measurements

    def encode_obs(self, image, py_measurements):
        """Encode args values to observation data (dict).

        Args:
            image (array): processed image after func pre_process()
            py_measurements (dict): measurement file

        Returns:
            dict: observation data
        """
        framestack = self._config["framestack"]
        assert framestack in [1, 2]
        prev_image = self._prev_image
        self._prev_image = image
        if prev_image is None:
            prev_image = image
        if framestack == 2:
            # image = np.concatenate([prev_image, image], axis=2)
            image = np.concatenate([prev_image, image])
        if not self._config["send_measurements"]:
            return image
        obs = (
            image,  # 'Vehicle number: ', vehicle_number,
            COMMAND_ORDINAL[py_measurements["next_command"]],
            [
                py_measurements["forward_speed"],
                py_measurements["distance_to_goal"]
            ])
        return obs

    def __del__(self):
        for actor in [
                self._collision_sensor.sensor,
                self._lane_invasion_sensor.sensor
        ]:
            if actor is not None:
                actor.destroy()
