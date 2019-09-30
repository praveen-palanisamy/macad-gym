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

import numpy as np  # linalg.norm is used
import GPUtil
from gym.spaces import Box, Discrete, Tuple, Dict
import pygame
import carla

from macad_gym.multi_actor_env import MultiActorEnv
from macad_gym import LOG_DIR
from macad_gym.core.sensors.utils import preprocess_image
from macad_gym.core.maps.nodeid_coord_map import TOWN01, TOWN02
# from macad_gym.core.sensors.utils import get_transform_from_nearest_way_point
from macad_gym.carla.reward import Reward
from macad_gym.core.sensors.hud import HUD
from macad_gym.viz.render import multi_view_render
from macad_gym.carla.scenarios import update_scenarios_parameter
# The following imports require carla to be imported already.
from macad_gym.core.sensors.camera_manager import CameraManager
from macad_gym.core.sensors.derived_sensors import LaneInvasionSensor
from macad_gym.core.sensors.derived_sensors import CollisionSensor
from macad_gym.core.controllers.keyboard_control import KeyboardControl
from macad_gym.carla.PythonAPI.agents.navigation.global_route_planner_dao \
    import GlobalRoutePlannerDAO

# The following imports depend on these paths being in sys path
# TODO: Fix this. This probably won't work after packaging/distribution
sys.path.append("src/macad_gym/carla/PythonAPI")
from macad_gym.core.maps.nav_utils import PathTracker  # noqa: E402
from macad_gym.carla.PythonAPI.agents.navigation.global_route_planner \
    import GlobalRoutePlanner  # noqa: E402
from macad_gym.carla.PythonAPI.agents.navigation.local_planner \
    import RoadOption  # noqa:E402

logger = logging.getLogger(__name__)
# Set this where you want to save image outputs (or empty string to disable)
CARLA_OUT_PATH = os.environ.get("CARLA_OUT", os.path.expanduser("~/carla_out"))
if CARLA_OUT_PATH and not os.path.exists(CARLA_OUT_PATH):
    os.makedirs(CARLA_OUT_PATH)

# Set this to the path of your Carla binary
SERVER_BINARY = os.environ.get(
    "CARLA_SERVER", os.path.expanduser("~/software/CARLA_0.9.4/CarlaUE4.sh"))

assert os.path.exists(SERVER_BINARY), "Make sure CARLA_SERVER environment" \
                                      " variable is set & is pointing to the" \
                                      " CARLA server startup script (Carla" \
                                      "UE4.sh). Refer to the README file/docs."

# TODO: Clean env & actor configs to have appropriate keys based on the nature
# of env
DEFAULT_MULTIENV_CONFIG = {
    "scenarios": "DEFAULT_SCENARIO_TOWN1",
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
        "send_measurements": False,
        "enable_planner": True,
        "sync_server": True
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
            "x_res": 84,
            "y_res": 84,
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

# Threshold to determine that the goal has been reached based on distance
DISTANCE_TO_GOAL_THRESHOLD = 0.5

# Threshold to determine that the goal has been reached based on orientation
ORIENTATION_TO_GOAL_THRESHOLD = math.pi / 4.0

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

WEATHERS = {
    0: carla.WeatherParameters.ClearNoon,
    1: carla.WeatherParameters.CloudyNoon,
    2: carla.WeatherParameters.WetNoon,
    3: carla.WeatherParameters.WetCloudyNoon,
    4: carla.WeatherParameters.MidRainyNoon,
    5: carla.WeatherParameters.HardRainNoon,
    6: carla.WeatherParameters.SoftRainNoon,
    7: carla.WeatherParameters.ClearSunset,
    8: carla.WeatherParameters.CloudySunset,
    9: carla.WeatherParameters.WetSunset,
    10: carla.WeatherParameters.WetCloudySunset,
    11: carla.WeatherParameters.MidRainSunset,
    12: carla.WeatherParameters.HardRainSunset,
    13: carla.WeatherParameters.SoftRainSunset,
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
except ImportError:
    logger.warning("\n Disabling RLlib support.", exc_info=True)
    pass


class MultiCarlaEnv(*MultiAgentEnvBases):
    def __init__(self, configs=None):
        """MACAD-Gym environment implementation.

        Provides a generic MACAD-Gym environment implementation that can be
        customized further to create new or variations of existing
        multi-agent learning environments. The environment settings, scenarios
        and the actors in the environment can all be configured using
        the `configs` dict.

        Args:
            configs (dict): Configuration for environment specified under the
                `env` key and configurations for each actor specified as dict
                under `actor`.
                Example:
                    >>> configs = {"env":{
                    "server_map":"/Game/Carla/Maps/Town05",
                    "discrete_actions":True,...},
                    "actor":{
                    "actor_id1":{"enable_planner":True,...},
                    "actor_id2":{"enable_planner":False,...}
                    }}
        """

        if configs is None:
            configs = DEFAULT_MULTIENV_CONFIG

        self._scenario_config = update_scenarios_parameter(
            configs)["scenarios"]
        self._env_config = configs["env"]
        self._actor_configs = configs["actors"]
        #: list of str: Supported values for `type` filed in `actor_configs`
        #: for actors than can be actively controlled
        self._supported_active_actor_types = [
            "vehicle_4W", "vehicle_2W", "pedestrian", "traffic_light"
        ]
        #: list of str: Supported values for `type` field in `actor_configs`
        #: for actors that are passive. Example: A camera mounted on a pole
        self._supported_passive_actor_types = ["camera"]

        # Set attributes as in gym's specs
        self.reward_range = (-float('inf'), float('inf'))
        self.metadata = {'render.modes': 'human'}

        # Belongs to env_config.
        self._server_map = self._env_config["server_map"]
        self._map = self._server_map.split("/")[-1]
        self._render = self._env_config["render"]
        self._framestack = self._env_config["framestack"]
        self._discrete_actions = self._env_config["discrete_actions"]
        self._squash_action_logits = self._env_config["squash_action_logits"]
        self._verbose = self._env_config["verbose"]
        self._render_x_res = self._env_config["render_x_res"]
        self._render_y_res = self._env_config["render_y_res"]
        self._x_res = self._env_config["x_res"]
        self._y_res = self._env_config["y_res"]
        self._use_depth_camera = False  # !!test
        self._cameras = {}
        self._sync_server = self._env_config["sync_server"]

        # self.config["server_map"] = "/Game/Carla/Maps/" + args.map

        # Initialize to be compatible with cam_manager to set HUD.
        pygame.font.init()  # for HUD
        self._hud = HUD(self._render_x_res, self._render_y_res)

        # Needed by macad_agents
        if self._discrete_actions:
            self.action_space = Dict({
                actor_id: Discrete(len(DISCRETE_ACTIONS))
                for actor_id in self._actor_configs.keys()
            })
        else:
            self.action_space = Dict({
                actor_id: Box(-1.0, 1.0, shape=(2, ))
                for actor_id in self._actor_configs.keys()
            })

        if self._use_depth_camera:
            self._image_space = Box(
                -1.0,
                1.0,
                shape=(self._y_res, self._x_res, 1 * self._framestack))
        else:  # Use RGB camera
            self._image_space = Box(
                0.0,
                255.0,
                shape=(self._y_res, self._x_res, 3 * self._framestack))
        if self._env_config["send_measurements"]:
            self.observation_space = Dict({
                actor_id: Tuple([
                    self._image_space,  # framestacked image
                    Discrete(len(COMMANDS_ENUM)),  # next_command
                    Box(-128.0, 128.0, shape=(2, ))
                ])  # forward_speed, dist to goal
                for actor_id in self._actor_configs.keys()
            })
        else:
            self.observation_space = Dict({
                actor_id: self._image_space
                for actor_id in self._actor_configs.keys()
            })

        #: Set appropriate node-id to coordinate mappings for Town01 or Town02.
        if self._map == "Town01":
            self.pos_coor_map = TOWN01
        else:
            self.pos_coor_map = TOWN02

        self._spec = lambda: None
        self._spec.id = "Carla-v0"
        self._server_port = None
        self._server_process = None
        self._client = None
        self._num_steps = {}
        self._total_reward = {}
        self._prev_measurement = {}
        self._prev_image = None
        self._episode_id_dict = {}
        self._measurements_file_dict = {}
        self._weather = None
        self._start_pos = {}  # Start pose for each actor
        self._end_pos = {}  # End pose for each actor
        self._start_coord = {}
        self._end_coord = {}
        self._last_obs = None
        self._image = None
        self._surface = None
        self._obs_dict = {}
        self._video = False
        self._previous_actions = {}
        self._previous_rewards = {}
        self._last_reward = {}
        self._agents = {}  # Dictionary of macad_agents with agent_id as key
        self._actors = {}  # Dictionary of actors with actor_id as key
        self._path_trackers = {}
        self._collisions = {}
        self._lane_invasions = {}
        self._scenario_map = {}
        self._load_scenario(self._scenario_config)
        self._done_dict = {}
        self._dones = set()  # Set of all done actor IDs

    @staticmethod
    def _get_free_tcp_port():
        s = socket.socket()
        s.bind(("", 0))  # Request the sys to provide a free port dynamically
        server_port = s.getsockname()[1]
        s.close()
        return server_port

    def _init_server(self):
        """Initialize carla server and client

        Returns:
            N/A
        """
        print("Initializing new Carla server...")
        # Create a new server process and start the client.
        # First find a port that is free and then use it in order to avoid
        # crashes due to:"...bind:Address already in use"
        self._server_port = MultiCarlaEnv._get_free_tcp_port()

        multigpu_success = False
        gpus = GPUtil.getGPUs()
        log_file = os.path.join(LOG_DIR,
                                "server_" + str(self._server_port) + ".log")

        if not self._render and (gpus is not None and len(gpus)) > 0:
            try:
                min_index = random.randint(0, len(gpus) - 1)
                for i, gpu in enumerate(gpus):
                    if gpu.load < gpus[min_index].load:
                        min_index = i
                # Check if vglrun is setup to launch sim on multipl GPUs
                if shutil.which("vglrun") is not None:
                    self._server_process = subprocess.Popen(
                        ("DISPLAY=:8 vglrun -d :7.{} {} {} -benchmark -fps=20"
                         " -carla-server -world-port={}"
                         " -carla-streaming-port=0".format(
                             min_index, SERVER_BINARY, self._server_map,
                             self._server_port)),
                        shell=True,
                        preexec_fn=os.setsid,
                        stdout=open(log_file, 'w'))

                # Else, run in headless mode and try the SDL CUDA hint
                else:
                    self._server_process = subprocess.Popen(
                        ("SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE={}"
                         " {} {} -benchmark -fps=20 -carla-server"
                         " -world-port={} -carla-streaming-port=0".format(
                             min_index, SERVER_BINARY, self._server_map,
                             self._server_port)),
                        shell=True,
                        preexec_fn=os.setsid,
                        stdout=open(log_file, 'w'))
            # TODO: Make the try-except style handling work with Popen
            # exceptions after launching the server procs are not caught
            except Exception as e:
                print(e)
            # Temporary soln to check if CARLA server proc started and wrote
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
                self._server_process = subprocess.Popen([
                    SERVER_BINARY, self._server_map, "-windowed", "-ResX=",
                    str(self._env_config["render_x_res"]), "-ResY=",
                    str(self._env_config["render_y_res"]), "-benchmark -fps=20"
                    "-carla-server",
                    "-carla-world-port={} -carla-streaming-port=0".format(
                        self._server_port)
                ],
                                                        preexec_fn=os.setsid,
                                                        stdout=open(
                                                            log_file, 'w'))
                print("Running simulation in single-GPU mode")
            except Exception as e:
                logger.debug(e)
                print("FATAL ERROR while launching server:", sys.exc_info()[0])

        live_carla_processes.add(os.getpgid(self._server_process.pid))

        # Start client
        self._client = None
        while self._client is None:
            try:
                self._client = carla.Client("localhost", self._server_port)
                self._client.set_timeout(2.0)
                self._client.get_server_version()
            except RuntimeError as re:
                if "timeout" not in str(re) and "time-out" not in str(re):
                    print("Could not connect to Carla server because:", re)
                self._client = None
        self._client.set_timeout(60.0)
        self.world = self._client.get_world()
        world_settings = self.world.get_settings()
        world_settings.synchronous_mode = self._sync_server
        self.world.apply_settings(world_settings)
        # Set the spectatator/server view if rendering is enabled
        if self._render and self._env_config.get("spectator_loc"):
            spectator = self.world.get_spectator()
            spectator_loc = carla.Location(*self._env_config["spectator_loc"])
            d = 6.4
            angle = 160  # degrees
            a = math.radians(angle)
            location = carla.Location(d * math.cos(a), d * math.sin(a),
                                      2.0) + spectator_loc
            spectator.set_transform(
                carla.Transform(location,
                                carla.Rotation(yaw=180 + angle, pitch=-15)))

        if self._env_config.get("enable_planner"):
            planner_dao = GlobalRoutePlannerDAO(self.world.get_map())
            self.planner = GlobalRoutePlanner(planner_dao)
            self.planner.setup()

    def _clean_world(self):
        """Destroy all actors cleanly before exiting

        Returns:
            N/A

        """

        for cam in self._cameras.values():
            if cam.sensor.is_alive:
                cam.sensor.destroy()

        for colli in self._collisions.values():
            if colli.sensor.is_alive:
                colli.sensor.destroy()
        for lane in self._lane_invasions.values():
            if lane.sensor.is_alive:
                lane.sensor.destroy()
        for actor in self._actors.values():
            if actor.is_alive:
                actor.destroy()
        # Clean-up any remaining vehicle in the world
        # for v in self.world.get_actors().filter("vehicle*"):
        #     if v.is_alive:
        #         v.destroy()
        #     assert (v not in self.world.get_actors())
        print("Cleaned-up the world...")

        self._cameras = {}
        self._actors = {}
        self._path_trackers = {}
        self._collisions = {}
        self._lane_invasions = {}

    def _clear_server_state(self):
        """Clear server process"""

        print("Clearing Carla server state")
        try:
            if self._client:
                self._client = None
        except Exception as e:
            print("Error disconnecting client: {}".format(e))
            pass
        if self._server_process:
            pgid = os.getpgid(self._server_process.pid)
            os.killpg(pgid, signal.SIGKILL)
            live_carla_processes.remove(pgid)
            self._server_port = None
            self._server_process = None

    def reset(self):
        """Reset the carla world, call _init_server()

        Returns:
            N/A

        """
        error = None
        for retry in range(RETRIES_ON_ERROR):
            try:
                if not self._server_process:
                    self.first_reset = True
                    self._init_server()
                return self._reset()
            except Exception as e:
                print("Error during reset: {}".format(traceback.format_exc()))
                print("reset(): Retry #: {}/{}".format(retry + 1,
                                                       RETRIES_ON_ERROR))
                self._clear_server_state()
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
        for cam in self._cameras.values():
            surface = cam._surface
            if surface is not None:
                self._display.blit(surface, (0, 0))
            pygame.display.flip()

    def _spawn_new_agent(self, actor_id):
        """Spawn an agent as per the blueprint at the given pose

        Args:
            blueprint: Blueprint of the actor. Can be a Vehicle or Pedestrian
            pose: carla.Transform object with location and rotation

        Returns:
            An instance of a subclass of carla.Actor. carla.Vehicle in the case
            of a Vehicle agent.

        """
        actor_type = self._actor_configs[actor_id].get("type", "vehicle_4W")
        if actor_type not in self._supported_active_actor_types:
            print("Unsupported actor type:{}. Using vehicle_4W as the type")
            actor_type = "vehicle_4W"

        if actor_type == "traffic_light":
            # Traffic lights already exist in the world & can't be spawned.
            # Find closest traffic light actor in world.actor_list and return
            from macad_gym.core.controllers import traffic_lights
            loc = carla.Location(self._start_pos[actor_id][0],
                                 self._start_pos[actor_id][1],
                                 self._start_pos[actor_id][2])
            rot = self.world.get_map().get_waypoint(
                loc, project_to_road=True).transform.rotation
            #: If yaw is provided in addition to (X, Y, Z), set yaw
            if len(self._start_pos[actor_id]) > 3:
                rot.yaw = self._start_pos[actor_id][3]
            transform = carla.Transform(loc, rot)
            self._actor_configs[actor_id]["start_transform"] = transform
            tls = traffic_lights.get_tls(self.world, transform, sort=True)
            return tls[0][0]  #: Return the key (carla.TrafficLight object) of
            #: closest match

        if actor_type == "pedestrian":
            blueprints = self.world.get_blueprint_library().filter("walker")

        elif actor_type == "vehicle_4W":
            blueprints = self.world.get_blueprint_library().filter("vehicle")
            # Further filter down to 4-wheeled vehicles
            blueprints = [
                b for b in blueprints
                if int(b.get_attribute("number_of_wheels")) == 4
            ]
        elif actor_type == "vehicle_2W":
            blueprints = self.world.get_blueprint_library().filter("vehicle")
            # Further filter down to 2-wheeled vehicles
            blueprints = [
                b for b in blueprints
                if int(b.get_attribute("number_of_wheels")) == 2
            ]

        blueprint = random.choice(blueprints)
        loc = carla.Location(
            x=self._start_pos[actor_id][0],
            y=self._start_pos[actor_id][1],
            z=self._start_pos[actor_id][2])
        rot = self.world.get_map().get_waypoint(
            loc, project_to_road=True).transform.rotation
        #: If yaw is provided in addition to (X, Y, Z), set yaw
        if len(self._start_pos[actor_id]) > 3:
            rot.yaw = self._start_pos[actor_id][3]
        transform = carla.Transform(loc, rot)
        self._actor_configs[actor_id]["start_transform"] = transform
        vehicle = None
        for retry in range(RETRIES_ON_ERROR - 1):
            vehicle = self.world.try_spawn_actor(blueprint, transform)
            if self._sync_server:
                self.world.tick()
                self.world.wait_for_tick()
            if vehicle is not None and vehicle.get_location().z > 0.0:
                break
            # Wait to see if spawn area gets cleared before retrying
            # time.sleep(0.5)
            # self._clean_world()
            print("spawn_actor: Retry#:{}/{}".format(retry + 1,
                                                     RETRIES_ON_ERROR))
        if vehicle is None:
            # Request a spawn one last time. Spit the error if it still fails
            vehicle = self.world.spawn_actor(blueprint, transform)
        return vehicle

    def _reset(self):
        """Reset the state of the actors.
        A "medium" reset is performed in which the existing actors are destroyed
        and the necessary actors are spawned into the environment without
        affecting other aspects of the environment.
        If the medium reset fails, a "hard" reset is performed in which
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
        if not self.first_reset:
            self._clean_world()
        self.first_reset = False

        weather_num = 0
        if "weather_distribution" in self._scenario_config:
            weather_num = \
                random.choice(self._scenario_config["weather_distribution"])
            if weather_num not in WEATHERS:
                weather_num = 0

        self.world.set_weather(WEATHERS[weather_num])

        self._weather = [
            self.world.get_weather().cloudyness,
            self.world.get_weather().precipitation,
            self.world.get_weather().precipitation_deposits,
            self.world.get_weather().wind_intensity
        ]

        for actor_id, actor_config in self._actor_configs.items():
            if self._done_dict.get(actor_id, None) is None or \
                    actor_id in self._dones:
                self._done_dict[actor_id] = True

            if self._done_dict.get(actor_id, False) is True:
                if actor_id in self._dones:
                    self._dones.remove(actor_id)
                if actor_id in self._collisions:
                    self._collisions[actor_id]._reset()
                if actor_id in self._lane_invasions:
                    self._lane_invasions[actor_id]._reset()
                if actor_id in self._path_trackers:
                    self._path_trackers[actor_id].reset()

                # Actor is not present in the simulation. Do a medium reset
                # by clearing the world and spawning the actor from scratch.
                # If the actor cannot be spawned, a hard reset is performed
                # which creates a new carla server instance

                # TODO: Once a unified (1 for all actors) scenario def is
                # implemented, move this outside of the foreach actor loop
                self._measurements_file_dict[actor_id] = None
                self._episode_id_dict[actor_id] = datetime.today().\
                    strftime("%Y-%m-%d_%H-%M-%S_%f")
                actor_config = self._actor_configs[actor_id]

                try:
                    self._actors[actor_id] = self._spawn_new_agent(actor_id)
                except RuntimeError as spawn_err:
                    self._done_dict[actor_id] = None
                    # Chain the exception & re-raise to be handled by the caller
                    # `self.reset()`
                    raise spawn_err from RuntimeError(
                        "Unable to spawn actor:{}".format(actor_id))

                if self._env_config["enable_planner"]:
                    self._path_trackers[actor_id] = PathTracker(
                        self.world, self.planner,
                        (self._start_pos[actor_id][0],
                         self._start_pos[actor_id][1],
                         self._start_pos[actor_id][2]),
                        (self._end_pos[actor_id][0],
                         self._end_pos[actor_id][1],
                         self._end_pos[actor_id][2]), self._actors[actor_id])

                # Spawn collision and lane sensors if necessary
                if actor_config["collision_sensor"] == "on":
                    collision_sensor = CollisionSensor(self._actors[actor_id],
                                                       0)
                    self._collisions.update({actor_id: collision_sensor})
                if actor_config["lane_sensor"] == "on":
                    lane_sensor = LaneInvasionSensor(self._actors[actor_id], 0)
                    self._lane_invasions.update({actor_id: lane_sensor})

                # Spawn cameras
                pygame.font.init()  # for HUD
                hud = HUD(self._env_config["x_res"], self._env_config["x_res"])
                camera_manager = CameraManager(self._actors[actor_id], hud)
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
                self._cameras.update({actor_id: camera_manager})

                self._start_coord.update({
                    actor_id: [
                        self._start_pos[actor_id][0] // 100,
                        self._start_pos[actor_id][1] // 100
                    ]
                })
                self._end_coord.update({
                    actor_id: [
                        self._end_pos[actor_id][0] // 100,
                        self._end_pos[actor_id][1] // 100
                    ]
                })

                print("Actor: {} start_pos_xyz(coordID): {} ({}), "
                      "end_pos_xyz(coordID) {} ({})".format(
                          actor_id, self._start_pos[actor_id],
                          self._start_coord[actor_id], self._end_pos[actor_id],
                          self._end_coord[actor_id]))

        print('New episode initialized with actors:{}'.format(
            self._actors.keys()))
        # TEMP: set traffic light to green for car2
        # tls = traffic_lights.get_tls(self.world,
        #                              self.actors["car2"].get_transform())
        # traffic_lights.set_tl_state(tls, carla.TrafficLightState.Green)

        for actor_id, cam in self._cameras.items():
            if self._done_dict.get(actor_id, False) is True:
                # TODO: Move the initialization value setting
                # to appropriate place
                # Set appropriate initial values
                self._last_reward[actor_id] = None
                self._total_reward[actor_id] = None
                self._num_steps[actor_id] = 0
                py_mt = self._read_observation(actor_id)
                py_measurement = py_mt
                self._prev_measurement[actor_id] = py_mt

                actor_config = self._actor_configs[actor_id]
                # Wait for the sensor (camera) actor to start streaming
                # Shouldn't take too long
                while cam.callback_count == 0:
                    if self._sync_server:
                        self.world.tick()
                        self.world.wait_for_tick()
                    pass
                if cam.image is None:
                    print("callback_count:", actor_id, ":", cam.callback_count)
                image = preprocess_image(cam.image, actor_config)
                obs = self._encode_obs(actor_id, image, py_measurement)
                self._obs_dict[actor_id] = obs

        return self._obs_dict

    def _load_scenario(self, scenario_parameter):
        self._scenario_map = {}
        # If config contains a single scenario, then use it,
        # if it's an array of scenarios,randomly choose one and init
        if isinstance(scenario_parameter, dict):
            scenario = scenario_parameter
        else:  # instance array of dict
            scenario = random.choice(scenario_parameter)

        self._scenario_map["max_steps"] = scenario["max_steps"]

        for actor_id, actor in scenario["actors"].items():
            if isinstance(actor["start"], int):
                self._start_pos[actor_id] = \
                    self.pos_coor_map[str(actor["start"])]
            else:
                self._start_pos[actor_id] = actor["start"]

            if isinstance(actor["end"], int):
                self._end_pos[actor_id] = \
                    self.pos_coor_map[str(actor["end"])]
            else:
                self._end_pos[actor_id] = actor["end"]

    def _encode_obs(self, actor_id, image, py_measurements):
        """Encode sensor and measurements into obs based on state-space config.

        Args:
            actor_id (str): Actor identifier
            image (array): processed image after func pre_process()
            py_measurements (dict): measurement file

        Returns:
            obs (dict): properly encoded observation data for each actor
        """
        assert self._framestack in [1, 2]
        prev_image = self._prev_image
        self._prev_image = image
        if prev_image is None:
            prev_image = image
        if self._framestack == 2:
            # image = np.concatenate([prev_image, image], axis=2)
            image = np.concatenate([prev_image, image])
        if not self._actor_configs[actor_id]["send_measurements"]:
            return image
        obs = (image, COMMAND_ORDINAL[py_measurements["next_command"]], [
            py_measurements["forward_speed"],
            py_measurements["distance_to_goal"]
        ])

        self._last_obs = obs
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

        if (not self._server_process) or (not self._client):
            raise RuntimeError("Cannot call step(...) before calling reset()")

        assert len(self._actors), "No actors exist in the environment. Either"\
                                  " the environment was not properly " \
                                  "initialized using`reset()` or all the " \
                                  "actors have exited. Cannot execute `step()`"

        if not isinstance(action_dict, dict):
            raise ValueError("`step(action_dict)` expected dict of actions. "
                             "Got {}".format(type(action_dict)))
        # Make sure the action_dict contains actions only for actors that
        # exist in the environment
        if not set(action_dict).issubset(set(self._actors)):
            raise ValueError("Cannot execute actions for non-existent actors."
                             " Received unexpected actor ids:{}".format(
                                 set(action_dict).difference(
                                     set(self._actors))))

        try:
            obs_dict = {}
            reward_dict = {}
            info_dict = {}

            for actor_id, action in action_dict.items():
                obs, reward, done, info = self._step(actor_id, action)
                obs_dict[actor_id] = obs
                reward_dict[actor_id] = reward
                self._done_dict[actor_id] = done
                if done:
                    self._dones.add(actor_id)
                info_dict[actor_id] = info
            self._done_dict["__all__"] = len(self._dones) == len(self._actors)
            # Find if any actor's config has render=True & render only for
            # that actor. NOTE: with async server stepping, enabling rendering
            # affects the step time & therefore MAX_STEPS needs adjustments
            render_required = [
                k for k, v in self._actor_configs.items()
                if v.get("render", False)
            ]
            if render_required:
                multi_view_render(obs_dict, [self._x_res, self._y_res],
                                  self._actor_configs)
            return obs_dict, reward_dict, self._done_dict, info_dict
        except Exception:
            print("Error during step, terminating episode early.",
                  traceback.format_exc())
            self._clear_server_state()

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

        if self._discrete_actions:
            action = DISCRETE_ACTIONS[int(action)]
        assert len(action) == 2, "Invalid action {}".format(action)
        if self._squash_action_logits:
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
        if self._verbose:
            print("steer", steer, "throttle", throttle, "brake", brake,
                  "reverse", reverse)

        config = self._actor_configs[actor_id]
        if config['manual_control']:
            clock = pygame.time.Clock()
            # pygame
            self._display = pygame.display.set_mode(
                (800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
            logger.debug('pygame started')
            controller = KeyboardControl(self, False)
            controller.actor_id = actor_id
            controller.parse_events(self, clock)
            # TODO: Is this _on_render() method necessary? why?
            self._on_render()
        elif config["auto_control"]:
            if getattr(self._actors[actor_id], 'set_autopilot', 0):
                self._actors[actor_id].set_autopilot()
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
                rotation = self._actors[actor_id].get_transform().rotation
                rotation.yaw += steer * 10.0
                x_dir = math.cos(math.radians(rotation.yaw))
                y_dir = math.sin(math.radians(rotation.yaw))

                self._actors[actor_id].apply_control(
                    carla.WalkerControl(
                        speed=3.0 * throttle,
                        direction=carla.Vector3D(x_dir, y_dir, 0.0)))

            # TODO: Change this if different vehicle types (Eg.:vehicle_4W,
            #  vehicle_2W, etc) have different control APIs
            elif "vehicle" in agent_type:
                self._actors[actor_id].apply_control(
                    carla.VehicleControl(
                        throttle=throttle,
                        steer=steer,
                        brake=brake,
                        hand_brake=hand_brake,
                        reverse=reverse))
        # Asynchronosly (one actor at a time; not all at once in a sync) apply
        # actor actions & perform a server tick after each actor's apply_action
        # if running with sync_server steps
        # NOTE: A distinction is made between "(A)Synchronous Environment" and
        # "(A)Synchronous (carla) server
        if self._sync_server:
            self.world.tick()
            self.world.wait_for_tick()

        # Process observations
        py_measurements = self._read_observation(actor_id)
        if self._verbose:
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

        # Compute reward
        config = self._actor_configs[actor_id]
        flag = config["reward_function"]
        cmpt_reward = Reward()
        reward = cmpt_reward.compute_reward(self._prev_measurement[actor_id],
                                            py_measurements, flag)

        self._previous_rewards[actor_id] = reward
        if self._total_reward[actor_id] is None:
            self._total_reward[actor_id] = reward
        else:
            self._total_reward[actor_id] += reward

        py_measurements["reward"] = reward
        py_measurements["total_reward"] = self._total_reward[actor_id]
        done = (self._num_steps[actor_id] > self._scenario_map["max_steps"]
                or py_measurements["next_command"] == "REACH_GOAL"
                or (config["early_terminate_on_collision"]
                    and collided_done(py_measurements)))
        py_measurements["done"] = done

        self._prev_measurement[actor_id] = py_measurements
        self._num_steps[actor_id] += 1

        if config["log_measurements"] and CARLA_OUT_PATH:
            # Write out measurements to file
            if not self._measurements_file_dict[actor_id]:
                self._measurements_file_dict[actor_id] = open(
                    os.path.join(
                        CARLA_OUT_PATH, "measurements_{}.json".format(
                            self._episode_id_dict[actor_id])), "w")
            self._measurements_file_dict[actor_id].\
                write(json.dumps(py_measurements))
            self._measurements_file_dict[actor_id].write("\n")
            if done:
                self._measurements_file_dict[actor_id].close()
                self._measurements_file_dict[actor_id] = None
                # if self.config["convert_images_to_video"] and\
                #  (not self.video):
                #    self.images_to_video()
                #    self.video = Trueseg_city_space

        original_image = self._cameras[actor_id].image
        config = self._actor_configs[actor_id]
        image = preprocess_image(original_image, config)

        return (self._encode_obs(actor_id, image, py_measurements), reward,
                done, py_measurements)

    def _read_observation(self, actor_id):
        """Read observation and return measurement.

        Args:
            actor_id (str): Actor identifier

        Returns:
            dict: measurement data.

        """
        cur = self._actors[actor_id]
        cur_config = self._actor_configs[actor_id]
        planner_enabled = cur_config["enable_planner"]
        if planner_enabled:
            dist = self._path_trackers[actor_id].get_distance_to_end()
            orientation_diff = self._path_trackers[actor_id].\
                get_orientation_difference_to_end_in_radians()
            commands = self.planner.plan_route(
                (cur.get_location().x, cur.get_location().y),
                (self._end_pos[actor_id][0], self._end_pos[actor_id][1]))
            if len(commands) > 0:
                next_command = ROAD_OPTION_TO_COMMANDS_MAPPING.get(
                    commands[0], "LANE_FOLLOW")
            elif dist <= DISTANCE_TO_GOAL_THRESHOLD and \
                    orientation_diff <= ORIENTATION_TO_GOAL_THRESHOLD:
                next_command = "REACH_GOAL"
            else:
                next_command = "LANE_FOLLOW"

            # DEBUG
            # self.path_trackers[actor_id].draw()
        else:
            next_command = "LANE_FOLLOW"

        collision_vehicles = self._collisions[actor_id].collision_vehicles
        collision_pedestrians = self._collisions[
            actor_id].collision_pedestrians
        collision_other = self._collisions[actor_id].collision_other
        intersection_otherlane = self._lane_invasions[actor_id].offlane
        intersection_offroad = self._lane_invasions[actor_id].offroad

        if next_command == "REACH_GOAL":
            distance_to_goal = 0.0
        elif planner_enabled:
            distance_to_goal = self._path_trackers[actor_id].\
                                   get_distance_to_end()
        else:
            distance_to_goal = -1

        distance_to_goal_euclidean = float(
            np.linalg.norm([
                self._actors[actor_id].get_location().x -
                self._end_pos[actor_id][0],
                self._actors[actor_id].get_location().y -
                self._end_pos[actor_id][1]
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
            "map": self._server_map,
            "start_coord": self._start_coord[actor_id],
            "end_coord": self._end_coord[actor_id],
            "current_scenario": self._scenario_map,
            "x_res": self._x_res,
            "y_res": self._y_res,
            "max_steps": self._scenario_map["max_steps"],
            "next_command": next_command,
            "previous_action": self._previous_actions.get(actor_id, None),
            "previous_reward": self._previous_rewards.get(actor_id, None)
        }

        return py_measurements

    def close(self):
        """Clean-up the world, clear server state & close the Env"""
        self._clean_world()
        self._clear_server_state()


def print_measurements(measurements):
    number_of_agents = len(measurements.non_player_agents)
    player_measurements = measurements.player_measurements
    message = "Vehicle at ({pos_x:.1f}, {pos_y:.1f}), "
    message += "{speed:.2f} km/h, "
    message += "Collision: {{vehicles={col_cars:.0f}, "
    message += "pedestrians={col_ped:.0f}, other={col_other:.0f}}}, "
    message += "{other_lane:.0f}% other lane, {offroad:.0f}% off-road, "
    message += "({agents_num:d} non-player macad_agents in the scene)"
    message = message.format(
        pos_x=player_measurements.transform.location.x,
        pos_y=player_measurements.transform.location.y,
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
    # TODO: Fix the default path to the config.json;Should work after packaging
    argparser.add_argument(
        '--config',
        default='src/macad_gym/carla/config.json',
        help='print debug information')

    argparser.add_argument(
        '--map', default='Town01', help='print debug information')

    args = argparser.parse_args()

    multi_env_config = json.load(open(args.config))
    env = MultiCarlaEnv(multi_env_config)

    for _ in range(2):
        obs = env.reset()

        total_reward_dict = {}
        action_dict = {}

        env_config = multi_env_config["env"]
        actor_configs = multi_env_config["actors"]
        for actor_id in actor_configs.keys():
            total_reward_dict[actor_id] = 0
            if env._discrete_actions:
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
            action_dict = get_next_actions(info, env._discrete_actions)
            for actor_id in total_reward_dict.keys():
                total_reward_dict[actor_id] += reward[actor_id]
            print(":{}\n\t".join(["Step#", "rew", "ep_rew", "done{}"]).format(
                i, reward, total_reward_dict, done))

        print("{} fps".format(i / (time.time() - start)))
