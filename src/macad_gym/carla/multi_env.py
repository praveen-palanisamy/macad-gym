"""
multi_env.py: Multi-actor environment interface for CARLA-Gym
Should support two modes of operation. See CARLA-Gym developer guide for
more information
__author__: @Praveen-Palanisamy
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import atexit
import shutil
from configparser import ConfigParser
from datetime import datetime
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

import gym
import numpy as np  # linalg.norm is used
import GPUtil
from gym.spaces import Box, Discrete, Tuple, Dict
import pygame
import carla
from gymnasium.utils import EzPickle
from path import Path
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

from macad_gym.core.constants import DEFAULT_MULTIENV_CONFIG, RETRIES_ON_ERROR, DISCRETE_ACTIONS, COMMANDS_ENUM, \
    ROAD_OPTION_TO_COMMANDS_MAPPING, DISTANCE_TO_GOAL_THRESHOLD, ORIENTATION_TO_GOAL_THRESHOLD, COMMAND_ORDINAL
from macad_gym.core.controllers.traffic import apply_traffic
from macad_gym import LOG_DIR
from macad_gym.core.world_objects.utils import preprocess_image
from macad_gym.core.maps.nodeid_coord_map import MAP_TO_COORDS_MAPPING
from macad_gym.core.maps.nodeid_coord_map import MAP_TO_COORDS_MAPPING

# from macad_gym.core.sensors.utils import get_transform_from_nearest_way_point
from macad_gym.carla.reward import Reward
from macad_gym.core.world_objects.hud import HUD
from macad_gym.core.utils.misc import sigmoid
from macad_gym.core.utils.scenario_config import Configuration
from macad_gym.viz.multi_view_renderer import MultiViewRenderer
from macad_gym.carla.scenarios import Scenarios

# The following imports require carla to be imported already.
from macad_gym.core.world_objects.camera_manager import CameraManager, CAMERA_TYPES, DEPTH_CAMERAS
from macad_gym.core.world_objects.sensors import LaneInvasionSensor
from macad_gym.core.world_objects.sensors import CollisionSensor
from macad_gym.core.controllers.keyboard_control import KeyboardControl
from macad_gym.carla.PythonAPI.agents.navigation.global_route_planner_dao import (  # noqa: E501
    GlobalRoutePlannerDAO,
)

# The following imports depend on these paths being in sys path
# TODO: Fix this. This probably won't work after packaging/distribution
sys.path.append("src/macad_gym/carla/PythonAPI")
from macad_gym.core.maps.nav_utils import PathTracker  # noqa: E402
from macad_gym.carla.PythonAPI.agents.navigation.global_route_planner import (  # noqa: E402, E501
    GlobalRoutePlanner,
)

logger = logging.getLogger(__name__)
# Set this where you want to save image outputs (or empty string to disable)
CARLA_OUT_PATH = os.environ.get("CARLA_OUT", os.path.expanduser("~/carla_out"))
if CARLA_OUT_PATH and not os.path.exists(CARLA_OUT_PATH):
    os.makedirs(CARLA_OUT_PATH)

# Set this to the path of your Carla binary
SERVER_BINARY = os.environ.get(
    "CARLA_SERVER", os.path.expanduser("~/software/CARLA_0.9.13/CarlaUE4.sh")
)

# Check if is using on Windows
IS_WINDOWS_PLATFORM = "win" in sys.platform

assert os.path.exists(SERVER_BINARY), (
    "Make sure CARLA_SERVER environment"
    " variable is set & is pointing to the"
    " CARLA server startup script (Carla"
    "UE4.sh). Refer to the README file/docs."
)

# TODO: Clean env & actor configs to have appropriate keys based on the nature
# from minigrid.core.actions import Actions
# from minigrid.core.constants import COLOR_NAMES, DIR_TO_VEC, TILE_PIXELS
# from minigrid.core.grid import Grid
# from minigrid.core.mission import MissionSpace
# from minigrid.core.world_object import Point, WorldObj
# TODO move constants to replicate minigrid
# TODO use obs and action spaces from gym





live_carla_processes = set()


def cleanup():
    print("Killing live carla processes", live_carla_processes)
    for pgid in live_carla_processes:
        if IS_WINDOWS_PLATFORM:
            # for Windows
            subprocess.call(["taskkill", "/F", "/T", "/PID", str(pgid)])
        else:
            # for Linux
            os.killpg(pgid, signal.SIGKILL)

    live_carla_processes.clear()


def termination_cleanup(*_):
    cleanup()
    sys.exit(0)


signal.signal(signal.SIGTERM, termination_cleanup)
signal.signal(signal.SIGINT, termination_cleanup)
atexit.register(cleanup)


def env(**kwargs):
    """Instantiate a PettingoZoo environment."""
    # TODO implement the AEC API interface as wrapper of immediately in the main env dependiong on the difference of the specification 
    env = MultiActorCarlaEnv(**kwargs)
    # env = SumoEnvironmentPZ(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)

class MultiActorCarlaEnv(gym.Env):
    def __init__(
        self,
        configs: dict = None,
        xml_config_path: str = None,
        maps_path: str = "./",
        render_mode: str = None,
        render_width: int = 800,
        render_height: int = 600,
        actor_render_width: int = 84,
        actor_render_height: int = 84,
        discrete_action_space: bool = True,
        max_steps: int = 500,
        sync_server: bool = True,
        fixed_delta_seconds: float = 0.05,
        verbose: bool = False
    ):
        """MACAD-Gym environment implementation.

        Provides a generic MACAD-Gym environment implementation that can be
        customized further to create new or variations of existing
        multi-agent learning environments. The environment settings, scenarios
        and the actors in the environment can all be configured using
        the `configs` dict.
        Examples:
            >>> env = MyMultiActorEnv(DEFAULT_MULTIENV_CONFIG)
            >>> obs = env.reset()
            >>> print(obs)
            {
                "car_0": [2.4, 1.6],
                "car_1": [3.4, -3.2],
                "camera_0": [0.0, 0.0, 10.0, 20.0. 30.0],
                "traffic_light_1": [0, 3, 5, 1],
            }
            >>> obs, rewards, dones, infos = env.step(
                action_dict={
                "car_0": 1, "car_1": 0, "camera_0": 1 "traffic_light_1": 2,
                })
    
            >>> print(rewards)
            {
                "car_0": 3,
                "car_1": -1,
                "camera_0": 1,
                "traffic_light_1": 0,
            }
            >>> print(dones)
            {
                "car_0": False,
                "car_1": True,
                "camera_0": False,
                "traffic_light_1": False,
                "__all__": False,
            }
        Args:
            configs (dict): Configuration for environment specified under the
                `env` key and configurations for each actor specified as dict
                under `actor`.
                Example:
                    >>> configs = {"env":{
                    "server_map":"/Game/Carla/Maps/Town05",
                    "discrete_actions":True,...},
                    "actors":{
                    "actor_id1":{"enable_planner":True,...},
                    "actor_id2":{"enable_planner":False,...}
                    }}
            xml_configs (str): filepath to an OPENscenario compatible xml file with scenarios configs.
                If both arguments are provided, `configs` overrides the XML scenarios.
        """
        assert configs is not None or xml_config_path is not None, "Missing configuration!"
        assert max_steps > 0,  "`max_steps` supports only values > 0."
        self.metadata = {"render_modes": "human", "render_fps": 60}

        # Load the configuration
        if xml_config_path is not None:
            configs = Configuration.parse_xml(xml_config_path).update(configs or {})
        else:
            configs = Configuration.parse(configs)
        self.scenario_configs = configs.scenarios
        self.actor_configs = configs.actors

        # Camera position is problematic for certain vehicles and even in autopilot they are prone to error
        self.exclude_hard_vehicles = False
        # list of str: Supported values for `type` filed in `actor_configs`
        self._supported_active_actor_types = ["vehicle_4W","vehicle_2W","pedestrian","traffic_light"]
        # Set attributes as in gym's specs
        self.reward_range = (-float("inf"), float("inf"))

        # Environment configuration variables
        self.server_maps_path = Path(maps_path)
        self.verbose = verbose
        self.render_mode = render_mode
        self.discrete_action_space = discrete_action_space
        self.max_steps = max_steps
        self._render_x_res = render_width
        self._render_y_res = render_height
        self._actor_render_x_res = actor_render_width
        self._actor_render_y_res = actor_render_height
        self._sync_server = sync_server
        self._fixed_delta_seconds = fixed_delta_seconds

        # Functionalities classes
        self._reward_policy = Reward()
        pygame.font.init()  # for HUD
        self._hud = HUD(self._render_x_res, self._render_y_res)

        # For manual_control
        manually_controlled = sum([actor_config.manual_control for actor_config in self.actor_configs.values()])
        if manually_controlled > 1:
            raise ValueError(f"At most one actor can be manually controlled, but {manually_controlled} were found with manual_control=True.")
        self._manual_control = manually_controlled > 0
        self._control_clock = None
        self._manual_controller = None
        self._manual_control_camera_manager = None

        # Render related
        self._actors_renderer = MultiViewRenderer(self._render_x_res, self._render_y_res)
        self._actors_renderer.set_surface_poses((self._actor_render_x_res, self._actor_render_y_res), self.actor_configs)

        # Action space (common for all actors)
        if self.discrete_action_space:
            n = len(DISCRETE_ACTIONS)
            self.action_space = Dict({
                actor_id: Discrete(n)
                for actor_id in self.actor_configs.keys()
            })
        else:
            self.action_space = Dict({
                actor_id: Box(-1.0, 1.0, shape=(2, ))
                for actor_id in self.actor_configs.keys()
            })

        # Output space of images after preprocessing
        self._image_space = Dict({
            actor_id: Box(0.0, 255.0, shape=(self._actor_render_y_res, self._actor_render_x_res, 1 * actor_config.framestack))
            if actor_config.camera_type in DEPTH_CAMERAS
            else Box(-1.0, 1.0, shape=(self._actor_render_y_res, self._actor_render_x_res, 3 * actor_config.framestack))
            for actor_id, actor_config in self.actor_configs.items()
        })
        # Observation space of output (for the individual actor)
        self.observation_space = Dict({
            actor_id: Tuple([
                self._image_space[actor_id],  # image
                Discrete(len(COMMANDS_ENUM)),  # next_command
                Box(-128.0, 128.0, shape=(2, )),  # forward_speed, dist to goal
            ]) if actor_config.send_measurements else self._image_space[actor_id]
            for actor_id, actor_config in self.actor_configs.items()
        })

        # Execution local variables
        self._server_port = None
        self._server_process = None
        self._client = None
        self._scenario_config = None  # Dictionary with current scenario map config
        self._num_steps = {}
        self._total_reward = {}
        self._prev_measurement = {}
        self._prev_image = {}  # Dictionary of images with actor_id as key to build a `framestack`
        self._episode_id_dict = {}
        self._measurements_file_dict = {}
        self._weather = None
        self._start_pos = {}  # Start pose for each actor
        self._end_pos = {}  # End pose for each actor
        self._start_coord = {}
        self._end_coord = {}
        self._last_obs = None
        self._surface = None
        self._video = False
        self._obs_dict = {}
        self._done_dict = {}
        self._previous_actions = {}
        self._previous_rewards = {}
        self._last_reward = {}
        self._npc_vehicles = []  # List of NPC vehicles
        self._npc_pedestrians = []  # List of NPC pedestrians
        self._agents = {}  # Dictionary of macad_agents with agent_id as key
        self._actors = {}  # Dictionary of actors with actor_id as key
        self._cameras = {}  # Dictionary of sensors with actor_id as key
        self._path_trackers = {}  # Dictionary of sensors with actor_id as key
        self._collisions = {}  # Dictionary of sensors with actor_id as key
        self._lane_invasions = {}  # Dictionary of sensors with actor_id as key

    @staticmethod
    def _get_tcp_port(port=0):
        """
        Get a free tcp port number
        :param port: (default 0) port number. When `0` it will be assigned a free port dynamically
        :return: a port number requested if free otherwise an unhandled exception would be thrown
        """
        s = socket.socket()
        s.bind(("", port))
        server_port = s.getsockname()[1]
        s.close()
        return server_port

    def _load_scenario(self):
        self._scenario_config = random.choice(list(self.scenario_configs.values()))

        for actor_id, actor in self._scenario_config.vehicles.items():
            self._start_pos[actor_id] = actor.start
            self._end_pos[actor_id] = actor.end

    def _init_server(self):
        """Initialize carla server and client

        Returns:
            N/A
        """
        print("Initializing new Carla server...")
        # Create a new server process and start the client.
        # First find a port that is free and then use it in order to avoid
        # crashes due to:"...bind:Address already in use"
        self._server_port = self._get_tcp_port()

        multigpu_success = False
        gpus = GPUtil.getGPUs()
        log_file = os.path.join(LOG_DIR, "server_" + str(self._server_port) + ".log")
        logger.info(f"1. Port: {self._server_port}\n"
                    f"2. Map: {self.server_maps_path / self._scenario_config.town}\n"
                    f"3. Binary: {SERVER_BINARY}")

        if not self.render_mode == "human" and (gpus is not None and len(gpus)) > 0:
            try:
                min_index = random.randint(0, len(gpus) - 1)
                for i, gpu in enumerate(gpus):
                    if gpu.load < gpus[min_index].load:
                        min_index = i
                # Check if vglrun is setup to launch sim on multipl GPUs
                if shutil.which("vglrun") is not None:
                    self._server_process = subprocess.Popen(
                        (f"DISPLAY=:8 vglrun -d :7.{min_index} {SERVER_BINARY} -benchmark -fps=20"
                         f" -carla-server -world-port={self._server_port} -carla-streaming-port=0"),
                        shell=True,
                        # for Linux
                        preexec_fn=None if IS_WINDOWS_PLATFORM else os.setsid,
                        # for Windows (not necessary)
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if IS_WINDOWS_PLATFORM else 0,
                        stdout=open(log_file, "w"),
                    )

                # Else, run in headless mode
                else:
                    # Since carla 0.9.12+ use -RenderOffScreen to start headlessly
                    # https://carla.readthedocs.io/en/latest/adv_rendering_options/
                    self._server_process = subprocess.Popen(
                        (  # 'SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE={} DISPLAY='
                         f'"{SERVER_BINARY}" -RenderOffScreen -benchmark -fps=20 -carla-server'
                         ' -world-port={self._server_port} -carla-streaming-port=0'
                        ),
                        shell=True,
                        # for Linux
                        preexec_fn=None if IS_WINDOWS_PLATFORM else os.setsid,
                        # for Windows (not necessary)
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if IS_WINDOWS_PLATFORM else 0,
                        stdout=open(log_file, "w"),
                    )
            # TODO: Make the try-except style handling work with Popen
            # exceptions after launching the server procs are not caught
            except Exception as e:
                print(e)
            # Temporary sol to check if CARLA server proc started and wrote
            # something to stdout which is the usual case during startup
            if os.path.isfile(log_file):
                multigpu_success = True
            else:
                multigpu_success = False

            if multigpu_success:
                print("Running sim servers in headless/multi-GPU mode")

        # Rendering mode and also a fallback if headless/multi-GPU doesn't work
        if multigpu_success is False:
            try:
                print("Using single gpu to initialize carla server")

                self._server_process = subprocess.Popen(
                    [
                        SERVER_BINARY,
                        "-windowed",
                        "-ResX=",
                        str(self._render_x_res),
                        "-ResY=",
                        str(self._render_y_res),
                        "-benchmark",
                        "-fps=20",
                        "-carla-server",
                        "-carla-rpc-port={}".format(self._server_port),
                        "-carla-streaming-port=0"
                    ],
                    # for Linux
                    preexec_fn=None if IS_WINDOWS_PLATFORM else os.setsid,
                    # for Windows (not necessary)
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if IS_WINDOWS_PLATFORM else 0,
                    stdout=open(log_file, "w"),
                )
                print("Running simulation in single-GPU mode")
            except Exception as e:
                logger.debug(e)
                print("FATAL ERROR while launching server:", sys.exc_info()[0])

        if IS_WINDOWS_PLATFORM:
            live_carla_processes.add(self._server_process.pid)
        else:
            live_carla_processes.add(os.getpgid(self._server_process.pid))

        # Start client
        self._client = None
        while self._client is None:
            try:
                self._client = carla.Client("localhost", self._server_port)
                # The socket establishment could take some time
                time.sleep(2)
                self._client.set_timeout(2.0)
                print("Client successfully connected to server, Carla-Server version: ",
                      self._client.get_server_version())
            except RuntimeError as re:
                if "timeout" not in str(re) and "time-out" not in str(re):
                    print("Could not connect to Carla server because:", re)
                self._client = None

        self._client.set_timeout(60.0)
        # load map using client api since 0.9.6+
        self._client.load_world(self.server_maps_path / self._scenario_config.town)
        self.world = self._client.get_world()
        world_settings = self.world.get_settings()
        # Synchronous_mode available with CARLA version>=0.9.6
        world_settings.synchronous_mode = self._sync_server
        if self._sync_server:
            # Set fixed_delta_seconds to have reliable physics between sim steps
            world_settings.fixed_delta_seconds = self._fixed_delta_seconds
        self.world.apply_settings(world_settings)
        # Set up traffic manager
        self._traffic_manager = self._client.get_trafficmanager()
        self._traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        self._traffic_manager.set_respawn_dormant_vehicles(True)
        self._traffic_manager.set_synchronous_mode(self._sync_server)
        # Set the spectator/server view if rendering is enabled
        if self.render_mode == "human":
            spectator = self.world.get_spectator()
            spectator_loc = carla.Location(*(list(self._start_pos.values())[0][:3] if len(self._start_pos)>0 else [0,0,0]))
            d = 6.4
            angle = 160  # degrees
            a = math.radians(angle)
            location = (carla.Location(d * math.cos(a), d * math.sin(a), 2.0) + spectator_loc)
            spectator.set_transform(
                carla.Transform(location, carla.Rotation(yaw=180 + angle, pitch=-15))
            )

        planner_dao = GlobalRoutePlannerDAO(self.world.get_map())
        self.planner = GlobalRoutePlanner(planner_dao)
        self.planner.setup()

    def _clean_world(self):
        """Destroy all actors cleanly before exiting

        Returns:
            N/A
        """
        for colli in self._collisions.values():
            if colli.sensor.is_alive:
                colli.sensor.destroy()
        for lane in self._lane_invasions.values():
            if lane.sensor.is_alive:
                lane.sensor.destroy()
        for actor in self._actors.values():
            if actor.is_alive:
                actor.destroy()
        for npc in self._npc_vehicles:
            npc.destroy()
        for npc in zip(*self._npc_pedestrians):
            npc[1].stop()  # stop controller
            npc[0].destroy()  # kill entity
        # Note: the destroy process for cameras is handled in camera_manager.py

        self._cameras = {}
        self._actors = {}
        self._npc_vehicles = []
        self._npc_pedestrians = []
        self._path_trackers = {}
        self._collisions = {}
        self._lane_invasions = {}

        print("Cleaned-up the world...")

    def _clear_server_state(self):
        """Clear server process"""
        print("Clearing Carla server state")
        try:
            if self._client:
                self._client = None
        except Exception as e:
            print("Error disconnecting client: {}".format(e))
        if self._server_process:
            if IS_WINDOWS_PLATFORM:
                subprocess.call(["taskkill", "/F", "/T", "/PID", str(self._server_process.pid)])
                live_carla_processes.remove(self._server_process.pid)
            else:
                pgid = os.getpgid(self._server_process.pid)
                os.killpg(pgid, signal.SIGKILL)
                live_carla_processes.remove(pgid)

            self._server_port = None
            self._server_process = None

    def _spawn_new_actor(self, actor_id):
        """Spawn an agent as per the blueprint at the given pose

        Args:
            blueprint: Blueprint of the actor. Can be a Vehicle or Pedestrian
            pose: carla.Transform object with location and rotation

        Returns:
            An instance of a subclass of carla.Actor. carla.Vehicle in the case
            of a Vehicle agent.

        """
        actor_type = self._scenario_config.vehicles[actor_id].type
        actor_model = self._scenario_config.vehicles[actor_id].model
        if actor_type not in self._supported_active_actor_types:
            print("Unsupported actor type:{}. Using vehicle_4W as the type")
            actor_type = "vehicle_4W"

        if actor_type == "traffic_light":
            # Traffic lights already exist in the world & can't be spawned.
            # Find closest traffic light actor in world.actor_list and return
            from macad_gym.core.controllers import traffic_lights

            loc = carla.Location(
                self._start_pos[actor_id][0],
                self._start_pos[actor_id][1],
                self._start_pos[actor_id][2],
            )
            rot = (self.world.get_map().get_waypoint(loc, project_to_road=True).transform.rotation)
            #: If yaw is provided in addition to (X, Y, Z), set yaw
            if len(self._start_pos[actor_id]) > 3:
                rot.yaw = self._start_pos[actor_id][3]
            transform = carla.Transform(loc, rot)
            tls = traffic_lights.get_tls(self.world, transform, sort=True)
            return tls[0][0]  #: Return the key (carla.TrafficLight object) of
            #: closest match

        if actor_type == "pedestrian":
            blueprints = self.world.get_blueprint_library().filter("walker.pedestrian.*" if actor_model is None else actor_model)

        elif actor_type == "vehicle_4W":
            blueprints = self.world.get_blueprint_library().filter("vehicle" if actor_model is None else actor_model)
            # Further filter down to 4-wheeled vehicles
            blueprints = [b for b in blueprints if int(b.get_attribute("number_of_wheels")) == 4]
            if self.exclude_hard_vehicles:
                blueprints = list(filter(lambda x: not
                    (x.id.endswith('microlino') or
                     x.id.endswith('carlacola') or
                     x.id.endswith('cybertruck') or
                     x.id.endswith('t2') or
                     x.id.endswith('sprinter') or
                     x.id.endswith('firetruck') or
                     x.id.endswith('ambulance')), blueprints))
        elif actor_type == "vehicle_2W":
            blueprints = self.world.get_blueprint_library().filter("vehicle" if actor_model is None else actor_model)
            # Further filter down to 2-wheeled vehicles
            blueprints = [b for b in blueprints if int(b.get_attribute("number_of_wheels")) == 2]

        blueprint = random.choice(blueprints)
        loc = carla.Location(
            x=self._start_pos[actor_id][0],
            y=self._start_pos[actor_id][1],
            z=self._start_pos[actor_id][2],
        )
        rot = (self.world.get_map().get_waypoint(loc, project_to_road=True).transform.rotation)
        #: If yaw is provided in addition to (X, Y, Z), set yaw
        if len(self._start_pos[actor_id]) > 3:
            rot.yaw = self._start_pos[actor_id][3]
        transform = carla.Transform(loc, rot)
        vehicle = None
        for retry in range(RETRIES_ON_ERROR):
            vehicle = self.world.try_spawn_actor(blueprint, transform)
            if self._sync_server:
                self.world.tick()
            if vehicle is not None and vehicle.get_location().z > 0.0:
                # Register it under traffic manager
                vehicle.set_autopilot(False, self._traffic_manager.get_port())
                # Register it under traffic manager
                # Walker vehicle type does not have autopilot. Use walker controller ai
                if actor_type == "pedestrian": #TODO review this
                    # vehicle.set_simulate_physics(False)
                    pass
                else:
                    vehicle.set_autopilot(False, self._traffic_manager.get_port())
                break
            # Wait to see if spawn area gets cleared before retrying
            # time.sleep(0.5)
            # self._clean_world()
            print("spawn_actor: Retry#:{}/{}".format(retry + 1, RETRIES_ON_ERROR))
        if vehicle is None:
            # Request a spawn one last time possibly raising the error
            vehicle = self.world.spawn_actor(blueprint, transform)
        return vehicle

    def _reset(self, clean_world=True):
        """Reset the state of the actors.
        A "soft" reset is performed in which the existing actors are destroyed
        and the necessary actors are spawned into the environment without
        affecting other aspects of the environment.
        If the "soft" reset fails, a "hard" reset is performed in which
        the environment's entire state is destroyed and a fresh instance of
        the server is created from scratch. Note that the "hard" reset is
        expected to take more time. In both of the reset modes ,the state/
        pose and configuration of all actors (including the sensor actors) are
        (re)initialized as per the actor configuration.

        Returns:
            dict: Dictionaries of observations for actors.

        Raises:
            RuntimeError: If spawning an actor at its initial state as per its'
            configuration fails (eg.: Due to collision with an existing object
            on that spot). This Error will be handled by the caller
            `self.reset()` which will perform a "hard" reset by creating
            a new server instance
        """

        self._done_dict["__all__"] = False
        if clean_world:
            self._clean_world()

        self.world.set_weather(random.choice(self._scenario_config.weathers))

        self._weather = [
            self.world.get_weather().cloudiness,
            self.world.get_weather().precipitation,
            self.world.get_weather().precipitation_deposits,
            self.world.get_weather().wind_intensity,
        ]

        # TODO to implement spawn also set(scenario_config.vehicles).diff(actor_configs) with proper behaviour
        for actor_id, actor_config in self.actor_configs.items():
            if self._done_dict.get(actor_id, True):

                self._measurements_file_dict[actor_id] = None
                self._episode_id_dict[actor_id] = datetime.today().strftime("%Y-%m-%d_%H-%M-%S_%f")
                actor_config = self.actor_configs[actor_id]

                # Try to spawn actor (soft reset) or fail and reinitialize the server before get back here
                try:
                    self._actors[actor_id] = self._spawn_new_actor(actor_id)
                except RuntimeError as spawn_err:
                    if actor_id in self._done_dict:
                        del self._done_dict[actor_id]
                    # Chain the exception & re-raise to be handled by the caller `self.reset()`
                    raise spawn_err from RuntimeError("Unable to spawn actor:{}".format(actor_id))

                self._path_trackers[actor_id] = PathTracker(
                    self.world,
                    self.planner,
                    (
                        self._start_pos[actor_id][0],
                        self._start_pos[actor_id][1],
                        self._start_pos[actor_id][2],
                    ),
                    (
                        self._end_pos[actor_id][0],
                        self._end_pos[actor_id][1],
                        self._end_pos[actor_id][2],
                    ),
                    self._actors[actor_id],
                )

                # Spawn collision and lane sensors if necessary
                if actor_config.collision_sensor:
                    collision_sensor = CollisionSensor(self._actors[actor_id], 0)
                    self._collisions.update({actor_id: collision_sensor})
                if actor_config.lane_sensor:
                    lane_sensor = LaneInvasionSensor(self._actors[actor_id], 0)
                    self._lane_invasions.update({actor_id: lane_sensor})

                # Spawn cameras
                pygame.font.init()  # for HUD
                hud = HUD(self._render_x_res, self._render_y_res)
                camera_manager = CameraManager(self._actors[actor_id], hud, record=actor_config.log_images)
                camera_manager.set_sensor(
                    CAMERA_TYPES[self.actor_configs[actor_id].camera_type].value-1,
                    self.actor_configs[actor_id].camera_position, notify=False
                )
                assert camera_manager.sensor.is_listening
                self._cameras.update({actor_id: camera_manager})

                # Manual Control
                if actor_config.manual_control:
                    # Set clock for sync
                    self.world.on_tick(self._hud.on_world_tick)
                    # Init objects for manual control
                    self._control_clock = pygame.time.Clock()
                    self._manual_controller = KeyboardControl(self, actor_config.auto_control)
                    self._manual_controller.actor_id = actor_id
                    self._manual_control_camera_manager = CameraManager(self._actors[actor_id], self._hud)
                    self._manual_control_camera_manager.set_sensor(CAMERA_TYPES["rgb"].value - 1, pos=2, notify=False)

                self._start_coord.update(
                    {
                        actor_id: [
                            self._start_pos[actor_id][0] // 100,
                            self._start_pos[actor_id][1] // 100,
                        ]
                    }
                )
                self._end_coord.update(
                    {
                        actor_id: [
                            self._end_pos[actor_id][0] // 100,
                            self._end_pos[actor_id][1] // 100,
                        ]
                    }
                )
                if self.verbose:
                    print(f"Actor: {actor_id}, "
                          f"start_pos_xyz(coordID): {self._start_pos[actor_id]} ({self._start_coord[actor_id]}), "
                          f"end_pos_xyz(coordID): {self._end_pos[actor_id]} ({self._end_coord[actor_id]})")

        print("New episode initialized with actors:{}".format(self._actors.keys()))

        self._npc_vehicles, self._npc_pedestrians = apply_traffic(
            self.world, self._traffic_manager,
            self._scenario_config.num_vehicles,
            self._scenario_config.num_pedestrians)

    def reset(self):
        """Reset the carla world, call _init_server()

        Returns:
            N/A
        """
        # World reset and new scenario selection if multiple are available
        self._load_scenario()

        for retry in range(RETRIES_ON_ERROR):
            try:
                if not self._server_process:
                    self._init_server()
                    self._reset(clean_world=False)
                else:
                    self._reset()
                break
            except Exception as e:
                print("Error during reset: {}".format(traceback.format_exc()))
                print("reset(): Retry #: {}/{}".format(retry + 1, RETRIES_ON_ERROR))
                self._clear_server_state()
                raise e

        # Set appropriate initial values for all actors
        for actor_id, actor_config in self.actor_configs.items():
            if self._done_dict.get(actor_id, True):
                self._last_reward[actor_id] = None
                self._total_reward[actor_id] = None
                self._num_steps[actor_id] = 0

                py_measurement = self._read_observation(actor_id)
                self._prev_measurement[actor_id] = py_measurement
                cam = self._cameras[actor_id]
                # Wait for the actor's camera sensor to start streaming (shouldn't take too long)
                while cam.callback_count == 0:
                    if self._sync_server:
                        self.world.tick()
                        # `wait_for_tick` is no longer needed, see https://github.com/carla-simulator/carla/pull/1803
                        # self.world.wait_for_tick()
                if cam.image is None:
                    print("callback_count:", actor_id, ":", cam.callback_count)
                obs = self._encode_obs(actor_id, cam.image, py_measurement)
                self._obs_dict[actor_id] = obs
                # Actor correctly reset
                self._done_dict[actor_id] = False

        return self._obs_dict

    def _read_observation(self, actor_id):
        """Read observation and return measurement.

        Args:
            actor_id (str): Actor identifier

        Returns:
            dict: measurement data.

        """
        cur = self._actors[actor_id]
        cur_config = self.actor_configs[actor_id]
        if cur_config.enable_planner:
            dist = self._path_trackers[actor_id].get_distance_to_end()
            orientation_diff = self._path_trackers[
                actor_id
            ].get_orientation_difference_to_end_in_radians()
            commands = self.planner.plan_route(
                (cur.get_location().x, cur.get_location().y),
                (self._end_pos[actor_id][0], self._end_pos[actor_id][1]),
            )
            if len(commands) > 0:
                next_command = ROAD_OPTION_TO_COMMANDS_MAPPING.get(
                    commands[0], "LANE_FOLLOW"
                )
            elif (dist <= DISTANCE_TO_GOAL_THRESHOLD
                  and orientation_diff <= ORIENTATION_TO_GOAL_THRESHOLD):
                next_command = "REACH_GOAL"
            else:
                next_command = "LANE_FOLLOW"

            # DEBUG
            # self.path_trackers[actor_id].draw()
        else:
            next_command = "LANE_FOLLOW"

        collision_vehicles = self._collisions[actor_id].collision_vehicles
        collision_pedestrians = self._collisions[actor_id].collision_pedestrians
        collision_other = self._collisions[actor_id].collision_other
        intersection_otherlane = self._lane_invasions[actor_id].offlane
        intersection_offroad = self._lane_invasions[actor_id].offroad

        if next_command == "REACH_GOAL":
            distance_to_goal = 0.0
        elif cur_config.enable_planner:
            distance_to_goal = self._path_trackers[actor_id].get_distance_to_end()
        else:
            distance_to_goal = -1

        distance_to_goal_euclidean = float(
            np.linalg.norm([
                self._actors[actor_id].get_location().x -
                self._end_pos[actor_id][0],
                self._actors[actor_id].get_location().y -
                self._end_pos[actor_id][1],
            ]))

        py_measurements = {
            "episode_id": self._episode_id_dict[actor_id],
            "step": self._num_steps[actor_id],
            "x": self._actors[actor_id].get_location().x,
            "y": self._actors[actor_id].get_location().y,
            "pitch": self._actors[actor_id].get_transform().rotation.pitch,
            "yaw": self._actors[actor_id].get_transform().rotation.yaw,
            "roll": self._actors[actor_id].get_transform().rotation.roll,
            "forward_speed": self._actors[actor_id].get_velocity().x,
            "distance_to_goal": distance_to_goal,
            "distance_to_goal_euclidean": distance_to_goal_euclidean,
            "collision_vehicles": collision_vehicles,
            "collision_pedestrians": collision_pedestrians,
            "collision_other": collision_other,
            "intersection_offroad": intersection_offroad,
            "intersection_otherlane": intersection_otherlane,
            "weather": self._weather,
            "map": self._scenario_config.town,
            "start_coord": self._start_coord[actor_id],
            "end_coord": self._end_coord[actor_id],
            "current_scenario": self._scenario_config,
            "x_res": self._render_x_res,
            "y_res": self._render_y_res,
            "max_steps": self.max_steps,
            "next_command": next_command,
            "previous_action": self._previous_actions.get(actor_id, None),
            "previous_reward": self._previous_rewards.get(actor_id, None),
        }

        return py_measurements

    def _encode_obs(self, actor_id, image, py_measurements):
        """Encode sensor and measurements into obs based on state-space config.

        Args:
            actor_id (str): Actor identifier
            image (array): original unprocessed image
            py_measurements (dict): measurement file

        Returns:
            obs (dict): properly encoded observation data for each actor
        """
        # Apply preprocessing
        actor_config = self.actor_configs[actor_id]
        image = preprocess_image(image, actor_config, resize=(self._actor_render_x_res, self._actor_render_y_res))
        # Stack frames
        prev_image = self._prev_image.get(actor_id, image)
        self._prev_image[actor_id] = image
        if actor_config.framestack == 2:
            # image = np.concatenate([prev_image, image], axis=2)
            image = np.concatenate([prev_image, image])
        # Structure the observation
        if not actor_config.send_measurements:
            return image
        obs = (
            image,
            COMMAND_ORDINAL[py_measurements["next_command"]],
            [py_measurements["forward_speed"], py_measurements["distance_to_goal"]],
        )

        self._last_obs = obs
        return obs

    def _decode_obs(self, actor_id, obs):
        """Decode actor observation into original image reversing the pre_process() operation.
        Args:
            actor_id (str): Actor identifier
            obs (dict): Properly encoded observation data of an actor

        Returns:
            image (array): Original actor camera view
        """
        if self.actor_configs[actor_id].send_measurements:
            obs = obs[0]
        # Reverse the processing operation
        if self.actor_configs[actor_id].camera_type in DEPTH_CAMERAS:
            img = np.tile(obs.swapaxes(0, 1), 3)
        else:
            img = obs.swapaxes(0, 1) * 128 + 128
        return img

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

        if self.discrete_action_space:
            action = DISCRETE_ACTIONS[int(action)]

        assert len(action) == 2, "Invalid action {}".format(action)
        if self.actor_configs[actor_id].squash_action_logits:
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
            print("steer", steer, "throttle", throttle, "brake", brake, "reverse", reverse)

        actor_config = self.actor_configs[actor_id]
        if actor_config.manual_control:
            self._control_clock.tick(self.metadata["render_fps"])
            self._manual_control_camera_manager._hud.tick(
                self.world,
                self._actors[actor_id],
                self._collisions[actor_id],
                self._control_clock,
            )
            self._manual_controller.parse_events(self, self._control_clock)

        elif actor_config.auto_control:
            if getattr(self._actors[actor_id], "set_autopilot", 0):
                self._actors[actor_id].set_autopilot(True, self._traffic_manager.get_port())
        else:
            agent_type = self._scenario_config.vehicles[actor_id].type
            # space of ped actors
            if agent_type == "pedestrian":
                # TODO: Add proper support for pedestrian actor according to action
                rotation = self._actors[actor_id].get_transform().rotation
                rotation.yaw += steer * 10.0
                x_dir = math.cos(math.radians(rotation.yaw))
                y_dir = math.sin(math.radians(rotation.yaw))

                self._actors[actor_id].apply_control(
                    carla.WalkerControl(
                        speed=3.0 * throttle,
                        direction=carla.Vector3D(x_dir, y_dir, 0.0),
                    )
                )
            # To cover vehicle_4W, vehicle_2W, etc
            elif "vehicle" in agent_type:
                self._actors[actor_id].apply_control(
                    carla.VehicleControl(
                        throttle=throttle,
                        steer=steer,
                        brake=brake,
                        hand_brake=hand_brake,
                        reverse=reverse,
                    )
                )
        # Asynchronosly (one actor at a time; not all at once in a sync) apply
        # actor actions & perform a server tick after each actor's apply_action
        # if running with sync_server steps
        # NOTE: A distinction is made between "(A)Synchronous Environment" and
        # "(A)Synchronous (carla) server"
        if self._sync_server:
            self.world.tick()
            # `wait_for_tick` is no longer needed, see https://github.com/carla-simulator/carla/pull/1803
            # self.world.wait_for_tick()

        # Process observations
        py_measurements = self._read_observation(actor_id)
        if self.verbose:
            print("Next command", py_measurements["next_command"])
        # Store previous action
        self._previous_actions[actor_id] = action
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

        # Compute done
        done = (self._num_steps[actor_id] > self.max_steps
                or py_measurements["next_command"] == "REACH_GOAL"
                or (actor_config.early_terminate_on_collision and collided_done(py_measurements)))
        py_measurements["done"] = done

        # Compute reward
        reward = self._reward_policy.compute_reward(
            self._prev_measurement[actor_id], py_measurements, actor_config.reward_function
        )

        self._previous_rewards[actor_id] = reward
        if self._total_reward[actor_id] is None:
            self._total_reward[actor_id] = reward
        else:
            self._total_reward[actor_id] += reward

        py_measurements["reward"] = reward
        py_measurements["total_reward"] = self._total_reward[actor_id]

        # End iteration updating parameters and logging
        self._prev_measurement[actor_id] = py_measurements
        self._num_steps[actor_id] += 1

        if actor_config.log_measurements and CARLA_OUT_PATH:
            # Write out measurements to file
            if not self._measurements_file_dict[actor_id]:
                self._measurements_file_dict[actor_id] = open(
                    os.path.join(
                        CARLA_OUT_PATH,
                        "measurements_{}.json".format(self._episode_id_dict[actor_id]),
                    ),
                    "w",
                )
            self._measurements_file_dict[actor_id].write(json.dumps(py_measurements))
            self._measurements_file_dict[actor_id].write("\n")
            if done:
                self._measurements_file_dict[actor_id].close()
                self._measurements_file_dict[actor_id] = None
                # if self.config["convert_images_to_video"] and\
                #  (not self.video):
                #    self.images_to_video()
                #    self.video = Trueseg_city_space
        original_image = self._cameras[actor_id].image

        return (
            self._encode_obs(actor_id, original_image, py_measurements),
            reward,
            done,
            py_measurements,
        )

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
            used to indicate env termination.
            info (dict): Info for each actor.

        Raises
            RuntimeError: If `step(...)` is called before calling `reset()`
            ValueError: If `action_dict` is not a dictionary of actions
            ValueError: If `action_dict` contains actions for nonexistent actor
        """

        if (not self._server_process) or (not self._client):
            raise RuntimeError("Cannot call step(...) before calling reset()")

        assert len(
            self._actors), ("No actors exist in the environment. Either"
                            " the environment was not properly "
                            "initialized using`reset()` or all the "
                            "actors have exited. Cannot execute `step()`")

        if not isinstance(action_dict, dict):
            raise ValueError(
                "`step(action_dict)` expected dict of actions. "
                "Got {}".format(type(action_dict))
            )
        # Make sure the action_dict contains actions only for actors that
        # exist in the environment
        if not set(action_dict).issubset(set(self._actors)):
            raise ValueError(
                "Cannot execute actions for non-existent actors."
                " Received unexpected actor ids:{}".format(
                    set(action_dict).difference(set(self._actors))
                )
            )

        try:
            obs_dict = {}
            reward_dict = {}
            info_dict = {}

            for actor_id, action in action_dict.items():
                obs, reward, done, info = self._step(actor_id, action)
                obs_dict[actor_id] = obs
                reward_dict[actor_id] = reward
                if not self._done_dict.get(actor_id, False):
                    self._done_dict[actor_id] = done
                info_dict[actor_id] = info
            self._done_dict["__all__"] = sum(self._done_dict.values()) >= len(self._actors)
            self.render()

            return obs_dict, reward_dict, self._done_dict, info_dict
        except Exception:
            print("Error during step, terminating episode early.", traceback.format_exc())
            self._clear_server_state()

    def render(self):
        """
        Render the pygame window.
        """
        # Render only the view of those actor with render=True
        if self.render_mode == "human" and sum([v.render for v in self.actor_configs.values()]) > 0:
            # Get images to render
            images = {}
            for actor_id, actor_config in self.actor_configs.items():
                if self.actor_configs[actor_id].render:
                    image = self._prev_image[actor_id]
                    # Reverse the processing operation
                    if self.actor_configs[actor_id].camera_type in DEPTH_CAMERAS:
                        image = np.tile(image.swapaxes(0, 1), 3)
                    else:
                        image = image.swapaxes(0, 1) * 128 + 128
                    images[actor_id] = image
            # Render
            if self._manual_control_camera_manager is not None:
                images["manual"] = self._manual_control_camera_manager.image
            self._actors_renderer.render(images)
            if self._manual_controller is None:
                self._actors_renderer.window_event_handler()

    def close(self):
        """Clean-up the world, clear server state & close the Env"""
        self._clean_world()
        self._clear_server_state()


# class SumoEnvironmentPZ(AECEnv, EzPickle):
#     """A wrapper for the SUMO environment that implements the AECEnv interface from PettingZoo.
# 
#     For more information, see https://pettingzoo.farama.org/api/aec/.
# 
#     The arguments are the same as for :py:class:`sumo_rl.environment.env.SumoEnvironment`.
#     """
# 
#     metadata = {"render.modes": ["human", "rgb_array"], "name": "sumo_rl_v0", "is_parallelizable": True}
# 
#     def __init__(self, **kwargs):
#         """Initialize the environment."""
#         EzPickle.__init__(self, **kwargs)
#         self._kwargs = kwargs
# 
#         self.seed()
#         self.env = SumoEnvironment(**self._kwargs)
# 
#         self.agents = self.env.ts_ids
#         self.possible_agents = self.env.ts_ids
#         self._agent_selector = agent_selector(self.agents)
#         self.agent_selection = self._agent_selector.reset()
#         # spaces
#         self.action_spaces = {a: self.env.action_spaces(a) for a in self.agents}
#         self.observation_spaces = {a: self.env.observation_spaces(a) for a in self.agents}
# 
#         # dicts
#         self.rewards = {a: 0 for a in self.agents}
#         self.terminations = {a: False for a in self.agents}
#         self.truncations = {a: False for a in self.agents}
#         self.infos = {a: {} for a in self.agents}
# 
#     def seed(self, seed=None):
#         """Set the seed for the environment."""
#         self.randomizer, seed = seeding.np_random(seed)
# 
#     def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
#         """Reset the environment."""
#         self.env.reset(seed=seed, options=options)
#         self.agents = self.possible_agents[:]
#         self.agent_selection = self._agent_selector.reset()
#         self.rewards = {agent: 0 for agent in self.agents}
#         self._cumulative_rewards = {agent: 0 for agent in self.agents}
#         self.terminations = {a: False for a in self.agents}
#         self.truncations = {a: False for a in self.agents}
#         self.infos = {agent: {} for agent in self.agents}
# 
#     def observation_space(self, agent):
#         """Return the observation space for the agent."""
#         return self.observation_spaces[agent]
# 
#     def action_space(self, agent):
#         """Return the action space for the agent."""
#         return self.action_spaces[agent]
# 
#     def observe(self, agent):
#         """Return the observation for the agent."""
#         obs = self.env.observations[agent].copy()
#         return obs
# 
#     def close(self):
#         """Close the environment and stop the SUMO simulation."""
#         self.env.close()
# 
#     def render(self, mode="human"):
#         """Render the environment."""
#         return self.env.render(mode)
# 
#     def save_csv(self, out_csv_name, episode):
#         """Save metrics of the simulation to a .csv file."""
#         self.env.save_csv(out_csv_name, episode)
# 
#     def step(self, action):
#         """Step the environment."""
#         if self.truncations[self.agent_selection] or self.terminations[self.agent_selection]:
#             return self._was_dead_step(action)
#         agent = self.agent_selection
#         if not self.action_spaces[agent].contains(action):
#             raise Exception(
#                 "Action for agent {} must be in Discrete({})."
#                 "It is currently {}".format(agent, self.action_spaces[agent].n, action)
#             )
# 
#         self.env._apply_actions({agent: action})
# 
#         if self._agent_selector.is_last():
#             self.env._run_steps()
#             self.env._compute_observations()
#             self.rewards = self.env._compute_rewards()
#             self.env._compute_info()
#         else:
#             self._clear_rewards()
# 
#         done = self.env._compute_dones()["__all__"]
#         self.truncations = {a: done for a in self.agents}
# 
#         self.agent_selection = self._agent_selector.next()
#         self._cumulative_rewards[agent] = 0
#         self._accumulate_rewards()

