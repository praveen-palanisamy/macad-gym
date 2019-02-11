"""
multi_env.py: Multi-actor environment interface for CARLA-Gym
Should support two modes of operation. See CARLA-Gym developer guide for
more information
__author__: PP
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import atexit
from datetime import datetime
import glob
import logging
import json
import os
import random
import signal
import subprocess
import sys
import time
import traceback
import socket
import math

import numpy as np  # linalg.norm is used
import GPUtil
from gym.spaces import Box, Discrete, Tuple
import pygame

from env.multi_actor_env import MultiActorEnv
from env.core.sensors.utils import preprocess_image
from env.core.maps.nodeid_coord_map import TOWN01, TOWN02
# from env.core.sensors.utils import get_transform_from_nearest_way_point
from env.carla.reward import Reward
from env.carla.agents.navigation.global_route_planner_dao \
    import GlobalRoutePlannerDAO
from env.carla.agents.navigation.global_route_planner \
    import GlobalRoutePlanner
from env.carla.agents.navigation.local_planner import RoadOption

from env.core.sensors.hud import HUD
from env.viz.render import multi_view_render

LOG_DIR = "logs"
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)
logging.basicConfig(filename=LOG_DIR + '/multi_env.log', level=logging.DEBUG)

try:
    import carla
except ImportError:
    try:
        # Find and use the egg file for the latest API version
        sys.path.append(
            glob.glob(f'**/**/PythonAPI/lib/carla-*{sys.version_info.major}.'
                      f'{sys.version_info.minor}-linux-x86_64.egg')[-1])
        import carla  # noqa: E402
    except IndexError:
        raise IndexError('CARLA PythonAPI egg file not found. Check the path')

# The following imports depend on carla. TODO: Can it be made better?
from env.core.sensors.camera_manager import CameraManager  # noqa: E402
from env.core.sensors.derived_sensors import LaneInvasionSensor  # noqa: E402
from env.core.sensors.derived_sensors import CollisionSensor  # noqa: E402
from env.core.controllers.keyboard_control import KeyboardControl  # noqa: E402

# Set this where you want to save image outputs (or empty string to disable)
CARLA_OUT_PATH = os.environ.get("CARLA_OUT", os.path.expanduser("~/carla_out"))
if CARLA_OUT_PATH and not os.path.exists(CARLA_OUT_PATH):
    os.makedirs(CARLA_OUT_PATH)

# Set this to the path of your Carla binary
SERVER_BINARY = os.environ.get(
    "CARLA_SERVER", os.path.expanduser("~/software/CARLA_0.9.3/CarlaUE4.sh"))

assert os.path.exists(SERVER_BINARY)

# TODO: Clean env & actor configs to have appropriate keys based on the nature
# of env
DEFAULT_MULTIENV_CONFIG = {
    "env": {
        "server_map": "/Game/Carla/Maps/Town01",
        "render": True,
        "render_x_res": 800,
        "render_y_res": 600,
        "x_res": 84,
        "y_res": 84,
        "framestack": 1,
        "discrete_actions": True,
        "squash_action_logits": False,
        "verbose": False,
        "use_depth_camera": False,
        "send_measurements": False
    },
    "actors": {
        "vehicle1": {
            "enable_planner": True,
            "render": True,  # Whether to render to screen or send to VFB
            "framestack": 1,  # note: only [1, 2] currently supported
            "convert_images_to_video": False,
            "early_terminate_on_collision": True,
            "verbose": False,
            "reward_function": "corl2017",
            "render_x_res": 800,
            "render_y_res": 600,
            "x_res": 84,
            "y_res": 84,
            "server_map": "/Game/Carla/Maps/Town01",
            "scenarios": "DEFAULT_SCENARIO_TOWN1",  # # no scenarios
            "use_depth_camera": False,
            "squash_action_logits": False,
            "manual_control": False,
            "auto_control": False,
            "camera_type": "rgb",
            "collision_sensor": "on",  # off
            "lane_sensor": "on",  # off
            "server_process": False,
            "send_measurements": False,
            "log_images": False,
            "log_measurements": False
        }
    }
}

# Carla planner commands
COMMANDS_ENUM = {
    0.0: "REACH_GOAL",
    5.0: "GO_STRAIGHT",
    4.0: "TURN_RIGHT",
    3.0: "TURN_LEFT",
    2.0: "LANE_FOLLOW",
}

# Mapping from string repr to one-hot encoding index to feed to the model
COMMAND_ORDINAL = {
    "REACH_GOAL": 0,
    "GO_STRAIGHT": 1,
    "TURN_RIGHT": 2,
    "TURN_LEFT": 3,
    "LANE_FOLLOW": 4,
}

ROAD_OPTION_TO_COMMANDS_MAPPING = {
    RoadOption.VOID: "REACH_GOAL",
    RoadOption.STRAIGHT: "GO_STRAIGHT",
    RoadOption.RIGHT: "TURN_RIGHT",
    RoadOption.LEFT: "TURN_LEFT",
    RoadOption.LANEFOLLOW: "LANE_FOLLOW",
}

# Number of retries if the server doesn't respond
RETRIES_ON_ERROR = 2

# Dummy Z coordinate to use when we only care about (x, y)
GROUND_Z = 22

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

live_carla_processes = set()


def cleanup():
    print("Killing live carla processes", live_carla_processes)
    for pgid in live_carla_processes:
        os.killpg(pgid, signal.SIGKILL)


def termination_cleanup(*_):
    cleanup()
    sys.exit(0)


signal.signal(signal.SIGTERM, termination_cleanup)
signal.signal(signal.SIGINT, termination_cleanup)
atexit.register(cleanup)

MultiAgentEnvBases = [MultiActorEnv]
try:
    from ray.rllib.env import MultiAgentEnv
    MultiAgentEnvBases.append(MultiAgentEnv)
except ImportError as err:
    logging.warning(err, "\n Disabling RLlib support.")
    pass


class MultiCarlaEnv(*MultiAgentEnvBases):
    def __init__(self, configs=DEFAULT_MULTIENV_CONFIG):
        """Carla environment implementation.

        The environment settings and scenarios are configure using env_config.
        Actors in the simulation that can be controlled are configured through
        the actor_configs (TODO: Separate env & actor configs).
        Args:
            configs (dict): Configuration for environment specified under the
            `env` key and configurations for each actor specified as dict under
            `actor`.
            Example:
                >>> configs = {
                "env": {"server_map": "/Game/Carla/Maps/Town02",
                "render": True,}, "actor": {"actor_id1":
                {"enable_planner": True},
                "actor_id2": {"enable_planner": False)}}}

        """
        self.env_config = configs["env"]
        self.actor_configs = configs["actors"]

        # Set attributes as in gym's specs
        self.reward_range = (-float('inf'), float('inf'))
        self.metadata = {'render.modes': 'human'}

        # Belongs to env_config.
        self.server_map = self.env_config["server_map"]
        self.city = self.server_map.split("/")[-1]
        self.render = self.env_config["render"]
        self.framestack = self.env_config["framestack"]
        self.discrete_actions = self.env_config["discrete_actions"]
        self.squash_action_logits = self.env_config["squash_action_logits"]
        self.verbose = self.env_config["verbose"]
        self.render_x_res = self.env_config["render_x_res"]
        self.render_y_res = self.env_config["render_y_res"]
        self.x_res = self.env_config["x_res"]
        self.y_res = self.env_config["y_res"]
        self.use_depth_camera = False  # !!test
        self.cameras = {}

        # self.config["server_map"] = "/Game/Carla/Maps/" + args.map

        # Initialize to be compatible with cam_manager to set HUD.
        pygame.font.init()  # for HUD
        self.hud = HUD(self.render_x_res, self.render_y_res)

        # Needed by agents
        if self.discrete_actions:
            self.action_space = Discrete(len(DISCRETE_ACTIONS))
        else:
            self.action_space = Box(-1.0, 1.0, shape=(2, ))

        if self.use_depth_camera:
            image_space = Box(
                -1.0, 1.0, shape=(self.y_res, self.x_res, 1 * self.framestack))
        elif self.env_config["send_measurements"]:
            image_space = Box(
                0.0,
                255.0,
                shape=(self.y_res, self.x_res, 3 * self.framestack))
            self.observation_space = Tuple([
                image_space,
                Discrete(len(COMMANDS_ENUM)),  # next_command
                Box(-128.0, 128.0, shape=(2, ))
            ])  # forward_speed, dist to goal
        else:
            self.observation_space = Box(
                0.0,
                255.0,
                shape=(self.y_res, self.x_res, 3 * self.framestack))

        # Set pos_coor map for Town01 or Town02.
        if self.city == "Town01":
            self.pos_coor_map = TOWN01
        else:
            self.pos_coor_map = TOWN02

        self._spec = lambda: None
        self._spec.id = "Carla-v0"
        self.server_port = None
        self.server_process = None
        self.client = None
        self.num_steps = {}
        self.total_reward = {}
        self.prev_measurement = {}
        self.prev_image = None
        self.episode_id_dict = {}
        self.measurements_file_dict = {}
        self.weather = None
        self.scenario = None
        self.start_pos = {}  # Start pose for each actor
        self.end_pos = {}  # End pose for each actor
        self.start_coord = {}
        self.end_coord = {}
        self.last_obs = None
        self.image = None
        self._surface = None
        self.obs_dict = {}
        self.video = False
        self.previous_actions = {}
        self.previous_rewards = {}
        self.last_reward = {}
        self.agents = {}  # Dictionary of agents with agent_id as key
        self.actors = {}  # Dictionary of actors with actor_id as key
        self.collisions = {}
        self.lane_invasions = {}
        self.scenario_map = {}
        self.done_dict = {}
        self.dones = set()  # Set of all done actor IDs

    def get_scenarios(self, choice):
        if choice == "DEFAULT_SCENARIO_TOWN1":
            from env.carla.scenarios import DEFAULT_SCENARIO_TOWN1
            return DEFAULT_SCENARIO_TOWN1
        elif choice == "DEFAULT_SCENARIO_TOWN1_2":
            from env.carla.scenarios import DEFAULT_SCENARIO_TOWN1_2
            return DEFAULT_SCENARIO_TOWN1_2
        elif choice == "DEFAULT_SCENARIO_TOWN2":
            from env.carla.scenarios import DEFAULT_SCENARIO_TOWN2
            return DEFAULT_SCENARIO_TOWN2
        elif choice == "TOWN1_STRAIGHT":
            from env.carla.scenarios import TOWN1_STRAIGHT
            return TOWN1_STRAIGHT
        elif choice == "CURVE_TOWN1":
            from env.carla.scenarios import CURVE_TOWN1
            return CURVE_TOWN1
        elif choice == "CURVE_TOWN2":
            from env.carla.scenarios import CURVE_TOWN2
            return CURVE_TOWN2
        elif choice == "DEFAULT_CURVE_TOWN1":
            from env.carla.scenarios import DEFAULT_CURVE_TOWN1
            return DEFAULT_CURVE_TOWN1
        # TODO: Combine the following INTERSECTION_TOWN3_* into a single config
        elif "INTERSECTION_TOWN3_CAR1" in choice:
            from env.carla.scenarios import INTERSECTION_TOWN3_CAR1
            return INTERSECTION_TOWN3_CAR1
        elif "INTERSECTION_TOWN3_CAR2" in choice:
            from env.carla.scenarios import INTERSECTION_TOWN3_CAR2
            return INTERSECTION_TOWN3_CAR2
        elif "INTERSECTION_TOWN3_PED1" in choice:
            from env.carla.scenarios import INTERSECTION_TOWN3_PED1
            return INTERSECTION_TOWN3_PED1
        elif "INTERSECTION_TOWN3_BIKE1" in choice:
            from env.carla.scenarios import INTERSECTION_TOWN3_BIKE1
            return INTERSECTION_TOWN3_BIKE1

    @staticmethod
    def get_free_tcp_port():
        s = socket.socket()
        s.bind(("", 0))  # Request the sys to provide a free port dynamically
        server_port = s.getsockname()[1]
        s.close()
        return server_port

    def init_server(self):
        """Initialize carla server and client

        Returns:
            N/A
        """
        print("Initializing new Carla server...")
        # Create a new server process and start the client.
        # First find a port that is free and then use it in order to avoid
        # crashes due to:"...bind:Address already in use"
        self.server_port = MultiCarlaEnv.get_free_tcp_port()

        multigpu_success = False
        gpus = GPUtil.getGPUs()
        log_file = os.path.join(LOG_DIR,
                                "server_" + str(self.server_port) + ".log")
        # TODO: Make the try-except style handling work with Popen
        if not self.render and (gpus is not None and len(gpus)) > 0:
            try:
                min_index = random.randint(0, len(gpus) - 1)
                for i, gpu in enumerate(gpus):
                    if gpu.load < gpus[min_index].load:
                        min_index = i
                self.server_process = subprocess.Popen(
                    ("DISPLAY=:8 vglrun -d :7.{} {} {} -benchmark -fps=10"
                     " -carla-server -carla-world-port={} -dumpmovie").format(
                         min_index, SERVER_BINARY, self.server_map,
                         self.server_port),
                    shell=True,
                    preexec_fn=os.setsid,
                    stdout=open(log_file, 'w'))
                multigpu_success = True
                print("Running simulation in multi-GPU mode")
            except Exception as e:
                print(e)

        # Single GPU and also fallback if multi-GPU doesn't work
        # TODO: Use env_config values for setting ResX, ResY params
        if multigpu_success is False:
            try:
                self.server_process = subprocess.Popen([
                    SERVER_BINARY, self.server_map, "-windowed", "-ResX=",
                    str(self.env_config["render_x_res"]), "-ResY=",
                    str(self.env_config["render_y_res"]), "-benchmark -fps=10"
                    "-carla-server", "-carla-world-port={} -dumpmovie".format(
                        self.server_port)
                ],
                                                       preexec_fn=os.setsid,
                                                       stdout=open(
                                                           log_file, 'w'))
                print("Running simulation in single-GPU mode")
            except Exception as e:
                logging.debug(e)
                print("FATAL ERROR while launching server:", sys.exc_info()[0])

        live_carla_processes.add(os.getpgid(self.server_process.pid))

        # Start client
        self.client = None
        while self.client is None:
            try:
                self.client = carla.Client("localhost", self.server_port)
                self.client.set_timeout(2.0)
                self.client.get_server_version()
            except RuntimeError as re:
                if "timeout" not in str(re):
                    print("Could not connect to Carla server because:", re)
                self.client = None
        self.client.set_timeout(60.0)
        self.world = self.client.get_world()
        # Set the spectatator/server view if rendering is enabled
        if self.render and self.env_config.get("spectator_loc"):
            spectator = self.world.get_spectator()
            spectator_loc = carla.Location(*self.env_config["spectator_loc"])
            d = 6.4
            angle = 160  # degrees
            a = math.radians(angle)
            location = carla.Location(d * math.cos(a), d * math.sin(a),
                                      2.0) + spectator_loc
            spectator.set_transform(
                carla.Transform(location,
                                carla.Rotation(yaw=180 + angle, pitch=-15)))

        if self.env_config.get("enable_planner"):
            planner_dao = GlobalRoutePlannerDAO(self.world.get_map())
            self.planner = GlobalRoutePlanner(planner_dao)
            self.planner.setup()

    def clean_world(self):
        """Destroy all actors cleanly before exiting

        Returns:
            N/A

        """

        for cam in self.cameras.values():
            if cam.sensor.is_alive:
                cam.sensor.destroy()

        for colli in self.collisions.values():
            if colli.sensor.is_alive:
                colli.sensor.destroy()
        for lane in self.lane_invasions.values():
            if lane.sensor.is_alive:
                lane.sensor.destroy()
        for actor in self.actors.values():
            if actor.is_alive:
                actor.destroy()
        # Clean-up any remaining vehicle in the world
        for v in self.world.get_actors().filter("vehicle*"):
            v.destroy()
            assert (v not in self.world.get_actors())
        time.sleep(0.2)
        print("Cleaned-up the world...")

        self.cameras = {}
        self.actors = {}
        self.collisions = {}
        self.lane_invasions = {}

    def clear_server_state(self):
        """Clear server process"""

        print("Clearing Carla server state")
        try:
            if self.client:
                self.client = None
        except Exception as e:
            print("Error disconnecting client: {}".format(e))
            pass
        if self.server_process:
            pgid = os.getpgid(self.server_process.pid)
            os.killpg(pgid, signal.SIGKILL)
            live_carla_processes.remove(pgid)
            self.server_port = None
            self.server_process = None

    def reset(self):
        """Reset the carla world, call init_server()

        Returns:
            N/A

        """
        error = None
        for retry in range(RETRIES_ON_ERROR):
            try:
                if not self.server_process:
                    self.init_server()
                return self._reset()
            except Exception as e:
                print("Error during reset: {}".format(traceback.format_exc()))
                print("reset(): Retry #: {}/{}".format(retry + 1,
                                                       RETRIES_ON_ERROR))
                self.clear_server_state()
                error = e
        raise error

    # TODO: Is this function required?
    # TODO: Thought: Run server in headless mode always. Use pygame win on
    # client when render=True
    def _on_render(self):
        """Render the pygame window.

        Args:

        Returns:
            N/A
        """
        for cam in self.cameras.values():
            surface = cam._surface
            if surface is not None:
                self._display.blit(surface, (0, 0))
            pygame.display.flip()

    def spawn_new_agent(self, actor_id):
        """Spawn an agent as per the blueprint at the given pose

        Args:
            blueprint: Blueprint of the actor. Can be a Vehicle or Pedestrian
            pose: carla.Transform object with location and rotation

        Returns:
            An instance of a subclass of carla.Actor. carla.Vehicle in the case
            of a Vehicle agent.

        """
        agent_type = self.actor_configs[actor_id].get("type", "vehicle")
        if agent_type not in ["pedestrian", "vehicle_2W", "vehicle_4W"]:
            print("Unsupported actor type:{}. Using vehicle_4W as the type")
            agent_type = "vehicle_4W"

        if agent_type == "pedestrian":
            blueprints = self.world.get_blueprint_library().filter('walker')

        elif agent_type == "vehicle_4W":
            blueprints = self.world.get_blueprint_library().filter('vehicle')
            # Further filter down to 4-wheeled vehicles
            blueprints = [
                b for b in blueprints
                if int(b.get_attribute('number_of_wheels')) == 4
            ]
        elif agent_type == "vehicle_2W":
            blueprints = self.world.get_blueprint_library().filter('vehicle')
            # Further filter down to 2-wheeled vehicles
            blueprints = [
                b for b in blueprints
                if int(b.get_attribute('number_of_wheels')) == 2
            ]

        blueprint = random.choice(blueprints)
        loc = carla.Location(
            x=self.start_pos[actor_id][0],
            y=self.start_pos[actor_id][1],
            z=self.start_pos[actor_id][2])
        rot = self.world.get_map().get_waypoint(
            loc, project_to_road=True).transform.rotation
        transform = carla.Transform(loc, rot)
        self.actor_configs[actor_id]["start_transform"] = transform
        vehicle = None
        for retry in range(RETRIES_ON_ERROR - 1):
            vehicle = self.world.try_spawn_actor(blueprint, transform)
            time.sleep(0.8)
            if vehicle is not None and vehicle.get_location().z > 0.0:
                break
            # Wait to see if spawn area gets cleared before retrying
            # time.sleep(0.5)
            # self.clean_world()
            print("spawn_actor: Retry#:{}/{}".format(retry + 1,
                                                     RETRIES_ON_ERROR))
        if vehicle is None:
            # Request a spawn one last time. Spit the error if it still fails
            vehicle = self.world.spawn_actor(blueprint, transform)
        return vehicle

    def _reset(self):
        """Initialize actors.

        Get poses for vehicle actors and spawn them.
        Spawn all sensors as needed.

        Returns:
            dict: observation dictionaries for actors.
        """

        self.done_dict["__all__"] = False
        # TODO: num_actors not equal num_vehicle. Fix it when other actors are
        # like pedestrians are added
        self.num_vehicle = len(self.actor_configs)
        self.weather = [
            self.world.get_weather().cloudyness,
            self.world.get_weather().precipitation,
            self.world.get_weather().precipitation_deposits,
            self.world.get_weather().wind_intensity
        ]

        # TODO: Setup env actors based on scenario description in env config
        for actor_id, actor_config in self.actor_configs.items():
            if self.done_dict.get(actor_id, None) is None:
                self.done_dict[actor_id] = True

            if self.done_dict.get(actor_id, False) is True:
                if actor_id in self.actors.keys():
                    # Actor is already in the simulation. Do a soft reset
                    transform = actor_config["start_transform"]
                    # TODO: Remove the default of using "vehicle" type once
                    # legacy support for config json/dicts without "type" is
                    # no more required
                    agent_type = actor_config.get("type", "vehicle")
                    if agent_type == "pedestrian":
                        self.actors[actor_id].apply_control(
                            carla.WalkerControl(speed=0.0))
                    elif agent_type == "vehicle":
                        self.actors[actor_id].apply_control(
                            carla.VehicleControl(
                                throttle=0.0,
                                steer=0.0,
                                brake=0.0,
                            ))
                    self.actors[actor_id].set_transform(transform)
                    if actor_id in self.dones:
                        self.dones.remove(actor_id)
                    if actor_id in self.collisions:
                        self.collisions[actor_id]._reset()
                    if actor_id in self.lane_invasions:
                        self.lane_invasions[actor_id]._reset()
                    # Wait until the actor is fully initialized. Otherwise,
                    # The control may be applied as the actor is being dropped
                    # into the scene
                    time.sleep(0.8)

                else:
                    # TODO: Move the following comments to method docstring
                    # Actor is not present in the simulation. Do a medium reset
                    # by clearing the world and spawning the actor from scratch.
                    # If the actor cannot be spawned, a hard reset is performed
                    # which creates a new carla server instance

                    # TODO: If scenario is same for all actors,move this outside
                    # of the foreach actor loop
                    self.measurements_file_dict[actor_id] = None
                    self.episode_id_dict[actor_id] = datetime.today().\
                        strftime("%Y-%m-%d_%H-%M-%S_%f")
                    actor_config = self.actor_configs[actor_id]
                    scenario = self.get_scenarios(actor_config["scenarios"])
                    # If config contains a single scenario, then use it,
                    # if it's an array of scenarios,randomly choose one and init
                    if isinstance(scenario, dict):
                        self.scenario_map.update({actor_id: scenario})
                    else:  # instance array of dict
                        self.scenario_map.\
                            update({actor_id: random.choice(scenario)})

                    self.scenario = self.scenario_map[actor_id]
                    if self.scenario.get("start_pos_id"):
                        # str(start_id).decode("utf-8") # for py2
                        s_id = str(self.scenario["start_pos_id"])
                        e_id = str(self.scenario["end_pos_id"])
                        self.start_pos[actor_id] = self.pos_coor_map[s_id]
                        self.end_pos[actor_id] = self.pos_coor_map[e_id]
                    elif self.scenario.get("start_pos_loc"):
                        self.start_pos[actor_id] = self.scenario[
                            "start_pos_loc"]
                        self.end_pos[actor_id] = self.scenario["end_pos_loc"]

                    self.actors[actor_id] = self.spawn_new_agent(actor_id)

                    if self.actors[actor_id] is None:
                        # Try to spawn for one last time. If it fails,
                        # a RuntimeExceptions is raised by  `world.spawn_actor`
                        # which is handled by the caller `self.reset()`
                        # vehicle = world.spawn_actor(blueprint, transform)
                        # OR:
                        raise RuntimeError(
                            "Unable to spawn actor:{}".format(actor_id))

                    print('Agent spawned at ',
                          self.actors[actor_id].get_location().x,
                          self.actors[actor_id].get_location().y,
                          self.actors[actor_id].get_location().z)

                    # Spawn collision and lane sensors if necessary
                    if actor_config["collision_sensor"] == "on":
                        collision_sensor = CollisionSensor(
                            self.actors[actor_id], 0)
                        self.collisions.update({actor_id: collision_sensor})
                    if actor_config["lane_sensor"] == "on":
                        lane_sensor = LaneInvasionSensor(
                            self.actors[actor_id], 0)
                        self.lane_invasions.update({actor_id: lane_sensor})

                    # Spawn cameras
                    pygame.font.init()  # for HUD
                    hud = HUD(self.env_config["x_res"],
                              self.env_config["x_res"])
                    camera_manager = CameraManager(self.actors[actor_id], hud)
                    if actor_config["log_images"]:
                        # TODO: The recording option should be part of config
                        # 1: Save to disk during runtime
                        # 2: save to memory first, dump to disk on exit
                        camera_manager.set_recording_option(1)

                    # TODO: Fix the hard-corded 0 id use sensor_type-> "camera"
                    # TODO: Make this consistent with keys
                    # in CameraManger's._sensors
                    camera_manager.set_sensor(0, notify=False)
                    assert (camera_manager.sensor.is_listening)
                    self.cameras.update({actor_id: camera_manager})

                    self.start_coord.update({
                        actor_id: [
                            self.start_pos[actor_id][0] // 100,
                            self.start_pos[actor_id][1] // 100
                        ]
                    })
                    self.end_coord.update({
                        actor_id: [
                            self.end_pos[actor_id][0] // 100,
                            self.end_pos[actor_id][1] // 100
                        ]
                    })

                    print(
                        "Actor: {} start_pos(coord): {} ({}), "
                        "end_pos(coord) {} ({})".format(
                            actor_id, self.start_pos[actor_id],
                            self.start_coord[actor_id], self.end_pos[actor_id],
                            self.end_coord[actor_id]))

        print('New episode initialized with actors:{}'.format(
            self.actors.keys()))
        # TEMP: set traffic light to green for car2
        # tl = self.actors["car2"].get_traffic_light()
        # tl.set_state(carla.TrafficLightState.Green)

        for actor_id, cam in self.cameras.items():
            if self.done_dict.get(actor_id, False) is True:
                # TODO: Move the initialization value setting
                # to appropriate place
                # Set appropriate initial values
                self.last_reward[actor_id] = None
                self.total_reward[actor_id] = None
                self.num_steps[actor_id] = 0
                py_mt = self._read_observation(actor_id)
                py_measurement = py_mt
                self.prev_measurement[actor_id] = py_mt

                actor_config = self.actor_configs[actor_id]
                # Wait for the sensor (camera) actor to start streaming
                # Shouldn't take too long
                while cam.callback_count == 0:
                    pass
                if cam.image is None:
                    print("callback_count:", actor_id, ":", cam.callback_count)
                image = preprocess_image(cam.image, actor_config)
                obs = self.encode_obs(actor_id, image, py_measurement)
                self.obs_dict[actor_id] = obs

        return self.obs_dict

    def encode_obs(self, actor_id, image, py_measurements):
        """Encode sensor and measurements into obs based on state-space config.

        Args:
            actor_id (str): Actor identifier
            image (array): processed image after func pre_process()
            py_measurements (dict): measurement file

        Returns:
            obs (dict): properly encoded observation data for each actor
        """
        assert self.framestack in [1, 2]
        prev_image = self.prev_image
        self.prev_image = image
        if prev_image is None:
            prev_image = image
        if self.framestack == 2:
            # image = np.concatenate([prev_image, image], axis=2)
            image = np.concatenate([prev_image, image])
        if not self.actor_configs[actor_id]["send_measurements"]:
            return image
        obs = (image, COMMAND_ORDINAL[py_measurements["next_command"]], [
            py_measurements["forward_speed"],
            py_measurements["distance_to_goal"]
        ])

        self.last_obs = obs
        return obs

    def step(self, action_dict):
        """Execute one environment step for the specified actors.

        Executes the provided action for the corresponding actors in the
        environment and returns the resulting environment observation, reward,
        done and info (measurements) for each of the actors. The step is
        performed asynchronously i.e. only for the specified actors and not
        necessarily for all actors in the environment.

        Args:
            action_dict (dict): Actions to be executed for each actor. Keys are
                agent_id strings, values are corresponding actions.

        Returns
            obs (dict): Observations for each actor.
            rewards (dict): Reward values for each actor. None for first step
            dones (dict): Done values for each actor. Special key "__all__" is
                set when all actors are done and the env terminates
            info (dict): Info for each actor.

        Raises
            RuntimeError: If `step(...)` is called before calling `reset()`
            ValueError: If `action_dict` is not a dictionary of actions
            ValueError: If `action_dict` contains actions for nonexistent actor
        """

        if (not self.server_process) or (not self.client):
            raise RuntimeError("Cannot call step(...) before calling reset()")

        assert len(self.actors), "No actors exist in the environment. Either" \
                                 " the environment was not properly " \
                                 "initialized using`reset()` or all the " \
                                 "actors have exited. Cannot execute `step()`."

        if not isinstance(action_dict, dict):
            raise ValueError("`step(action_dict)` expected dict of actions. "
                             "Got {}".format(type(action_dict)))
        # Make sure the action_dict contains actions only for actors that
        # exist in the environment
        if not set(action_dict).issubset(set(self.actors)):
            raise ValueError("Cannot execute actions for non-existent actors."
                             " Received unexpected actor ids:{}".format(
                                 set(action_dict).difference(set(
                                     self.actors))))

        try:
            obs_dict = {}
            reward_dict = {}
            info_dict = {}

            for actor_id, action in action_dict.items():
                obs, reward, done, info = self._step(actor_id, action)
                obs_dict[actor_id] = obs
                reward_dict[actor_id] = reward
                self.done_dict[actor_id] = done
                if done:
                    self.dones.add(actor_id)
                info_dict[actor_id] = info
            self.done_dict["__all__"] = len(self.dones) == len(self.actors)
            multi_view_render(obs_dict, [self.x_res, self.y_res],
                              self.actor_configs)
            return obs_dict, reward_dict, self.done_dict, info_dict
        except Exception:
            print("Error during step, terminating episode early.",
                  traceback.format_exc())
            self.clear_server_state()

    def _step(self, actor_id, action):
        """Perform the actual step in the CARLA environment

        Applies control to `actor_id` based on `action`, process measurements,
        compute the rewards and terminal state info (dones).

        Args:
            actor_id(str): Actor identifier
            action: Actions to be executed for the actor.

        Returns
            obs (obs_space): Observation for the actor whose id is actor_id.
            reward (float): Reward for actor. None for first step
            done (bool): Done value for actor.
            info (dict): Info for actor.
        """

        if self.discrete_actions:
            action = DISCRETE_ACTIONS[int(action)]
        assert len(action) == 2, "Invalid action {}".format(action)
        if self.squash_action_logits:
            forward = 2 * float(sigmoid(action[0]) - 0.5)
            throttle = float(np.clip(forward, 0, 1))
            brake = float(np.abs(np.clip(forward, -1, 0)))
            steer = 2 * float(sigmoid(action[1]) - 0.5)
        else:
            throttle = float(np.clip(action[0], 0, 0.6))
            brake = float(np.abs(np.clip(action[0], -1, 0)))
            steer = float(np.clip(action[1], -1, 1))
        reverse = False
        hand_brake = False
        if self.verbose:
            print("steer", steer, "throttle", throttle, "brake", brake,
                  "reverse", reverse)

        config = self.actor_configs[actor_id]
        if config['manual_control']:
            clock = pygame.time.Clock()
            # pygame
            self._display = pygame.display.set_mode(
                (800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
            logging.debug('pygame started')
            controller = KeyboardControl(self, False)
            controller.actor_id = actor_id
            controller.parse_events(self, clock)
            # TODO: Is this _on_render() method necessary? why?
            self._on_render()
        elif config["auto_control"]:
            self.actors[actor_id].set_autopilot()
        else:
            # TODO: Planner based on waypoints.
            # cur_location = self.actor_list[i].get_location()
            # dst_location = carla.Location(x = self.end_pos[i][0],
            # y = self.end_pos[i][1], z = self.end_pos[i][2])
            # cur_map = self.world.get_map()
            # next_point_transform = get_transform_from_nearest_way_point(
            # cur_map, cur_location, dst_location)
            # the point with z = 0, and the default z of cars are 40
            # next_point_transform.location.z = 40
            # self.actor_list[i].set_transform(next_point_transform)

            agent_type = config.get("type", "vehicle")
            # TODO: Add proper support for pedestrian actor according to action
            # space of ped actors
            if agent_type == "pedestrian":
                self.actors[actor_id].apply_control(
                    carla.WalkerControl(speed=2.0 + throttle))
            elif agent_type == "vehicle":
                self.actors[actor_id].apply_control(
                    carla.VehicleControl(
                        throttle=throttle,
                        steer=steer,
                        brake=brake,
                        hand_brake=hand_brake,
                        reverse=reverse))

        # Process observations
        py_measurements = self._read_observation(actor_id)
        if self.verbose:
            print("Next command", py_measurements["next_command"])
        # Store previous action
        self.previous_actions[actor_id] = action
        if type(action) is np.ndarray:
            py_measurements["action"] = [float(a) for a in action]
        else:
            py_measurements["action"] = action
        py_measurements["control"] = {
            "steer": steer,
            "throttle": throttle,
            "brake": brake,
            "reverse": reverse,
            "hand_brake": hand_brake,
        }

        # Compute reward
        config = self.actor_configs[actor_id]
        flag = config["reward_function"]
        cmpt_reward = Reward()
        reward = cmpt_reward.compute_reward(self.prev_measurement[actor_id],
                                            py_measurements, flag)

        self.previous_rewards[actor_id] = reward
        if self.total_reward[actor_id] is None:
            self.total_reward[actor_id] = reward
        else:
            self.total_reward[actor_id] += reward

        py_measurements["reward"] = reward
        py_measurements["total_reward"] = self.total_reward[actor_id]
        done = (self.num_steps[actor_id] > self.scenario["max_steps"]
                or py_measurements["next_command"] == "REACH_GOAL"
                or (config["early_terminate_on_collision"]
                    and collided_done(py_measurements)))
        py_measurements["done"] = done

        self.prev_measurement[actor_id] = py_measurements
        self.num_steps[actor_id] += 1

        if config["log_measurements"] and CARLA_OUT_PATH:
            # Write out measurements to file
            if not self.measurements_file_dict[actor_id]:
                self.measurements_file_dict[actor_id] = open(
                    os.path.join(
                        CARLA_OUT_PATH, "measurements_{}.json".format(
                            self.episode_id_dict[actor_id])), "w")
            self.measurements_file_dict[actor_id].\
                write(json.dumps(py_measurements))
            self.measurements_file_dict[actor_id].write("\n")
            if done:
                self.measurements_file_dict[actor_id].close()
                self.measurements_file_dict[actor_id] = None
                # if self.config["convert_images_to_video"] and\
                #  (not self.video):
                #    self.images_to_video()
                #    self.video = Trueseg_city_space

        original_image = self.cameras[actor_id].image
        config = self.actor_configs[actor_id]
        image = preprocess_image(original_image, config)

        return (self.encode_obs(actor_id, image, py_measurements), reward,
                done, py_measurements)

    def _read_observation(self, actor_id):
        """Read observation and return measurement.

        Args:
            actor_id (str): Actor identifier

        Returns:
            dict: measurement data.

        """
        cur = self.actors[actor_id]
        cur_config = self.actor_configs[actor_id]
        planner_enabled = cur_config["enable_planner"]
        if planner_enabled:
            commands = self.planner.plan_route(
                (cur.get_location().x, cur.get_location().y),
                (self.end_pos[actor_id][0], self.end_pos[actor_id][1]))
            if len(commands) > 0:
                next_command = ROAD_OPTION_TO_COMMANDS_MAPPING.get(
                    commands[0], "LANE_FOLLOW")
            else:
                next_command = "LANE_FOLLOW"
        else:
            next_command = "LANE_FOLLOW"

        collision_vehicles = self.collisions[actor_id].collision_vehicles
        collision_pedestrians = self.collisions[actor_id].collision_pedestrians
        collision_other = self.collisions[actor_id].collision_other
        intersection_otherlane = self.lane_invasions[actor_id].offlane
        intersection_offroad = self.lane_invasions[actor_id].offroad

        if next_command == "REACH_GOAL":
            distance_to_goal = 0.0  # avoids crash in planner
        elif planner_enabled:
            distance_to_goal = self.planner.get_shortest_path_distance(
                (cur.get_location().x, cur.get_location().y),
                (self.end_pos[actor_id][0], self.end_pos[actor_id][1])) / 100
        else:
            distance_to_goal = -1

        distance_to_goal_euclidean = float(
            np.linalg.norm([
                self.actors[actor_id].get_location().x -
                self.end_pos[actor_id][0],
                self.actors[actor_id].get_location().y -
                self.end_pos[actor_id][1]
            ]) / 100)

        py_measurements = {
            "episode_id": self.episode_id_dict[actor_id],
            "step": self.num_steps[actor_id],
            "x": self.actors[actor_id].get_location().x,
            "y": self.actors[actor_id].get_location().y,
            "pitch": self.actors[actor_id].get_transform().rotation.pitch,
            "yaw": self.actors[actor_id].get_transform().rotation.yaw,
            "roll": self.actors[actor_id].get_transform().rotation.roll,
            "forward_speed": self.actors[actor_id].get_velocity().x,
            "distance_to_goal": distance_to_goal,
            "distance_to_goal_euclidean": distance_to_goal_euclidean,
            "collision_vehicles": collision_vehicles,
            "collision_pedestrians": collision_pedestrians,
            "collision_other": collision_other,
            "intersection_offroad": intersection_offroad,
            "intersection_otherlane": intersection_otherlane,
            "weather": self.weather,
            "map": self.server_map,
            "start_coord": self.start_coord[actor_id],
            "end_coord": self.end_coord[actor_id],
            "current_scenario": self.scenario_map[actor_id],
            "x_res": self.x_res,
            "y_res": self.y_res,
            "num_vehicles": self.scenario_map[actor_id]["num_vehicles"],
            "num_pedestrians": self.scenario_map[actor_id]["num_pedestrians"],
            "max_steps": self.scenario["max_steps"],
            "next_command": next_command,
            "previous_action": self.previous_actions.get(actor_id, None),
            "previous_reward": self.previous_rewards.get(actor_id, None)
        }

        return py_measurements


def print_measurements(measurements):
    number_of_agents = len(measurements.non_player_agents)
    player_measurements = measurements.player_measurements
    message = "Vehicle at ({pos_x:.1f}, {pos_y:.1f}), "
    message += "{speed:.2f} km/h, "
    message += "Collision: {{vehicles={col_cars:.0f}, "
    message += "pedestrians={col_ped:.0f}, other={col_other:.0f}}}, "
    message += "{other_lane:.0f}% other lane, {offroad:.0f}% off-road, "
    message += "({agents_num:d} non-player agents in the scene)"
    message = message.format(
        pos_x=player_measurements.transform.location.x / 100,  # cm -> m
        pos_y=player_measurements.transform.location.y / 100,
        speed=player_measurements.forward_speed,
        col_cars=player_measurements.collision_vehicles,
        col_ped=player_measurements.collision_pedestrians,
        col_other=player_measurements.collision_other,
        other_lane=100 * player_measurements.intersection_otherlane,
        offroad=100 * player_measurements.intersection_offroad,
        agents_num=number_of_agents)
    print(message)


def sigmoid(x):
    x = float(x)
    return np.exp(x) / (1 + np.exp(x))


def collided_done(py_measurements):
    """Define the main episode termination criteria"""
    m = py_measurements
    collided = (m["collision_vehicles"] > 0 or m["collision_pedestrians"] > 0
                or m["collision_other"] > 0)
    return bool(collided)  # or m["total_reward"] < -100)


def get_next_actions(measurements, is_discrete_actions):
    """Get/Update next action, work with way_point based planner.

    Args:
        measurements (dict): measurement data.
        is_discrete_actions (bool): whether use discrete actions

    Returns:
        dict: action_dict, dict of len-two integer lists.
    """
    action_dict = {}
    for actor_id, meas in measurements.items():
        m = meas
        command = m["next_command"]
        if command == "REACH_GOAL":
            action_dict[actor_id] = 0
        elif command == "GO_STRAIGHT":
            action_dict[actor_id] = 3
        elif command == "TURN_RIGHT":
            action_dict[actor_id] = 6
        elif command == "TURN_LEFT":
            action_dict[actor_id] = 5
        elif command == "LANE_FOLLOW":
            action_dict[actor_id] = 3
        # Test for discrete actions:
        if not is_discrete_actions:
            action_dict[actor_id] = [1, 0]
    return action_dict


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '--scenario', default='3', help='print debug information')

    argparser.add_argument(
        '--config',
        default='env/carla/config.json',
        help='print debug information')

    argparser.add_argument(
        '--map', default='Town01', help='print debug information')

    args = argparser.parse_args()

    multi_env_config = json.load(open(args.config))
    env = MultiCarlaEnv(multi_env_config)

    for _ in range(2):
        obs = env.reset()
        total_vehicle = env.num_vehicle

        total_reward_dict = {}
        action_dict = {}

        env_config = multi_env_config["env"]
        actor_configs = multi_env_config["actors"]
        for actor_id in actor_configs.keys():
            total_reward_dict[actor_id] = 0
            if env.discrete_actions:
                action_dict[actor_id] = 3  # Forward
            else:
                action_dict[actor_id] = [1, 0]  # test values

        start = time.time()
        i = 0
        done = {"__all__": False}
        while not done["__all__"]:
            # while i < 20:  # TEST
            i += 1
            obs, reward, done, info = env.step(action_dict)
            action_dict = get_next_actions(info, env.discrete_actions)
            for actor_id in total_reward_dict.keys():
                total_reward_dict[actor_id] += reward[actor_id]
            print(":{}\n\t".join(["Step#", "rew", "ep_rew", "done{}"]).format(
                i, reward, total_reward_dict, done))

            time.sleep(0.1)

        print("{} fps".format(i / (time.time() - start)))

    # Clean actors in world
    env.clean_world()

    # env.camera_list.save_images_to_disk()
