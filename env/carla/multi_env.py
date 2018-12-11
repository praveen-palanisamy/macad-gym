"""
multi_env.py: Multi-actor environment interface for CARLA-Gym
Should support two modes of operation. See CARLA-Gym developer guide for
more information
__author__: PP, BP
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

import numpy as np  # linalg.norm is used
import GPUtil
from gym.spaces import Box, Discrete, Tuple
import pygame

from env.multi_actor_env import MultiActorEnv
from env.core.sensors.utils import preprocess_image
# from env.core.sensors.utils import get_transform_from_nearest_way_point
from env.carla.reward import Reward
from env.carla.carla.planner.planner import Planner
from env.core.sensors.camera_list import CameraList
from env.core.sensors.hud import HUD
sys.path.append(
    glob.glob(f'**/**/PythonAPI/lib/carla-*{sys.version_info.major}.'
              f'{sys.version_info.minor}-linux-x86_64.egg')[0])
import carla  # noqa: E402
# The following import depend on carla. TODO: Can it be made better?
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
    "CARLA_SERVER", os.path.expanduser("~/software/CARLA_0.9.1/CarlaUE4.sh"))

assert os.path.exists(SERVER_BINARY)

DEFAULT_MULTIENV_CONFIG = {
    "env": {
        "server_map": "/Game/Carla/Maps/Town01",
        "render": True,
        "render_x_res": 800,
        "render_y_res": 600,
        "x_res": 80,
        "y_res": 80,
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
            "discrete_actions": False,
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

# Number of retries if the server doesn't respond
RETRIES_ON_ERROR = 5

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


def termination_cleanup(ig, nore):
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
    logging.WARN(err, "\n Disabling RLlib support.")
    pass


class MultiCarlaEnv(*MultiAgentEnvBases):
    def __init__(self, configs=DEFAULT_MULTIENV_CONFIG):
        """Carla environment implementation.

        The environment settings and scenarios are configure using env_config.
        Actors in the simulation that can be controlled are configured through
        the actor_configs (TODO: Separate env & actor configs).
        Args:
            A list of config files for actors.

        """
        tmp = iter(configs.values())
        self.env_config = next(tmp)
        self.actor_configs = next(tmp)

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
        self.camera_list = CameraList(CARLA_OUT_PATH)
        self.planner = Planner(self.city)  # A* based navigation planner

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
            self.pos_coor_map = json.load(
                open("env/carla/POS_COOR/pos_cordi_map_town1.txt"))
        else:
            self.pos_coor_map = json.load(
                open("env/carla/POS_COOR/pos_cordi_map_town2.txt"))

        self._spec = lambda: None
        self._spec.id = "Carla-v0"
        self.server_port = None
        self.server_process = None
        self.client = None
        self.num_steps = {}
        self.total_reward = {}
        self.prev_measurement = None
        self.prev_image = None
        self.episode_id = None
        self.measurements_file = None
        self.weather = None
        self.scenario = None
        self.start_pos = None
        self.end_pos = None
        self.start_coord = None
        self.end_coord = None
        self.last_obs = None
        self.image = None
        self._surface = None
        self.obs_dict = {}
        self.video = False
        self.previous_actions = {}
        self.previous_rewards = {}
        self.last_reward = {}
        self.actors = {}  # Dictionary of actors with actor_id as key
        self.collisions = {}
        self.lane_invasions = {}
        self.scenario_map = {}

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

    def init_server(self):
        """Initialize carla server and client

        Returns:
            N/A
        """
        print("Initializing new Carla server...")
        # Create a new server process and start the client.
        self.server_port = random.randint(10000, 60000)

        gpus = GPUtil.getGPUs()
        if not self.render and (gpus is not None and len(gpus)) > 0:
            min_index = random.randint(0, len(gpus) - 1)
            for i, gpu in enumerate(gpus):
                if gpu.load < gpus[min_index].load:
                    min_index = i
            # TODO: Use env_config values for setting ResX, ResY params
            self.server_process = subprocess.Popen(
                ("DISPLAY=:8 vglrun -d :7.{} {} " + self.server_map +
                 "-windowed -ResX={} -ResY={} -carla-server"
                 "-carla-world-port={}").format(min_index, SERVER_BINARY, 800,
                                                600, self.server_port),
                shell=True,
                preexec_fn=os.setsid,
                stdout=subprocess.PIPE)
        else:
            self.server_process = subprocess.Popen([
                SERVER_BINARY, self.server_map, "-windowed", "-ResX=800",
                "-ResY=600", "-benchmark -fps=10"
                "-carla-server", "-carla-world-port={}".format(
                    self.server_port)
            ],
                                                   preexec_fn=os.setsid,
                                                   stdout=subprocess.PIPE)

        live_carla_processes.add(os.getpgid(self.server_process.pid))

        time.sleep(10)  # Wait for CARLA server to initialize

        # Start client
        self.client = carla.Client("localhost", self.server_port)

    def clean_world(self):
        """Destroy all actors cleanly before exiting

        Returns:
            N/A

        """

        for cam in self.camera_list.cam_list.values():
            cam.sensor.destroy()
        for actor in self.actors.values():
            actor.destroy()
        for colli in self.collisions.values():
            colli.sensor.destroy()
        for lane in self.lane_invasions.values():
            lane.sensor.destroy()

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
        for _ in range(RETRIES_ON_ERROR):
            try:
                if not self.server_process:
                    self.init_server()
                    print('Server is intiated.')
                return self._reset()
            except Exception as e:
                print("Error during reset: {}".format(traceback.format_exc()))
                self.clear_server_state()
                error = e
        raise error

    def _on_render(self, i=0):
        """Render the pygame window.

        Args:
            i (int): the index of list self.actor_list.

        Returns:
            N/A
        """
        # 0 instead of i, easy test
        i = 0
        surface = self.camera_list.cam_list[i]._surface
        if surface is not None:
            self._display.blit(surface, (0, 0))
        pygame.display.flip()

    def _reset(self):
        """Initialize actors.

        Get poses for vehicle actors and spawn them.
        Spawn all sensors as needed.

        Returns:
            dict: observation dictionaries for actors.
        """

        # TODO: num_actors not equal num_vehicle. Fix it when other actors are
        # like pedestrians are added
        self.num_vehicle = len(self.actor_configs)
        self.prev_measurement = None
        self.prev_image = None
        self.episode_id = datetime.today().strftime("%Y-%m-%d_%H-%M-%S_%f")
        self.measurements_file = None
        self.start_pos = {}  # Start pose for each actor
        self.end_pos = {}  # End pose for each actor
        self.start_coord = {}
        self.end_coord = {}
        self.py_measurement = {}
        self.prev_measurement = {}
        self.obs = []

        world = self.client.get_world()
        self.world = world
        self.weather = [
            world.get_weather().cloudyness,
            world.get_weather().precipitation,
            world.get_weather().precipitation_deposits,
            world.get_weather().wind_intensity
        ]

        for actor_id, actor_config in self.actor_configs.items():
            actor_config = self.actor_configs[actor_id]
            scenario = self.get_scenarios(actor_config["scenarios"])
            # If config contains a single scenario, then use it,
            # if it's an array of scenarios, randomly choose one and init
            if isinstance(scenario, dict):
                self.scenario_map.update({actor_id: scenario})
            else:  # instance array of dict
                self.scenario_map.update({actor_id: random.choice(scenario)})

            self.scenario = self.scenario_map[actor_id]
            # str(start_id).decode("utf-8") # for py2
            s_id = str(self.scenario["start_pos_id"])
            e_id = str(self.scenario["end_pos_id"])
            self.start_pos[actor_id] = self.pos_coor_map[s_id]
            self.end_pos[actor_id] = self.pos_coor_map[e_id]

            blueprints = world.get_blueprint_library().filter('vehicle')
            # Further filter down to 4-wheeled vehicles
            blueprints = [
                b for b in blueprints
                if int(b.get_attribute('number_of_wheels')) == 4
            ]
            blueprint = random.choice(blueprints)
            transform = carla.Transform(
                carla.Location(
                    x=self.start_pos[actor_id][0],
                    y=self.start_pos[actor_id][1],
                    z=self.start_pos[actor_id][2]),
                carla.Rotation(pitch=0, yaw=0, roll=0))
            print('spawning vehicle %r with %d wheels' %
                  (blueprint.id, blueprint.get_attribute('number_of_wheels')))

            # Spawn actors
            vehicle = world.try_spawn_actor(blueprint, transform)
            print('vehicle at ',
                  vehicle.get_location().x,
                  vehicle.get_location().y,
                  vehicle.get_location().z)
            self.actors.update({actor_id: vehicle})

            # Spawn collision and lane sensors if necessary
            if actor_config["collision_sensor"] == "on":
                collision_sensor = CollisionSensor(vehicle, 0)
                self.collisions.update({actor_id: collision_sensor})
            if actor_config["lane_sensor"] == "on":
                lane_sensor = LaneInvasionSensor(vehicle, 0)
                self.lane_invasions.update({actor_id: lane_sensor})

            # Spawn cameras
            pygame.font.init()  # for HUD
            hud = HUD(self.env_config["render_x_res"],
                      self.env_config["render_y_res"])
            camera_manager = CameraManager(self.actors[actor_id], hud)
            if actor_config["log_images"]:
                # TODO: The recording option should be part of config
                # 1: Save to disk during runtime
                # 2: save to memory first, dump to disk on exit
                camera_manager.set_recording_option(1)

            # TODO: Fix the hard-corded 0 id
            camera_manager.set_sensor(0, notify=False)
            self.camera_list.cam_list.update({actor_id: camera_manager})

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
                "Actor: {} start_pos(coord): {} ({}), end_pos(coord) {} ({})".
                format(actor_id, self.start_pos[actor_id],
                       self.start_coord[actor_id], self.end_pos[actor_id],
                       self.end_coord[actor_id]))

        time.sleep(0.5)  # Small wait for the server to spawn all the actors
        print('Environment initialized with requested actors.')
        # TODO: CameraList class looks unnecessary. Cleanup.
        for actor_id, cam in self.camera_list.cam_list.items():
            # TODO: Move the initialization value setting to appropriate place
            # Set appropriate initial values
            self.last_reward[actor_id] = None
            self.total_reward[actor_id] = None
            self.num_steps[actor_id] = 0
            py_mt = self._read_observation(actor_id)
            self.py_measurement[actor_id] = py_mt
            self.prev_measurement[actor_id] = py_mt

            actor_config = self.actor_configs[actor_id]
            image = preprocess_image(cam.image, actor_config)
            obs = self.encode_obs(actor_id, image,
                                  self.py_measurement[actor_id])
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
            ValueError: If `action_dict` contains actions for non-existent actor
        """

        if (not self.server_process) or (not self.client):
            raise RuntimeError("Cannot call step(...) before calling reset()")

        assert len(self.actors), "No actors exist in the environment. Either" \
                                 " the environment was not properly " \
                                 "initialized using`reset()` or all the" \
                                 " actors have exited. Cannot execute `step()`."

        if not isinstance(action_dict, dict):
            raise ValueError("`step(action_dict)` expected dict of actions. "
                             "Got {}".format(type(action_dict)))
        # Make sure the action_dict contains actions only for actors that
        # exist in the environment
        # if not set(action_dict).issubset(set(self.actors)):
        #    raise ValueError("Cannot execute actions for non-existent actors."
        #                     " Received unexpected actor ids:{}".format(
        #                         set(action_dict).difference(set(
        #                             self.actors))))

        try:
            obs_dict = {}
            reward_dict = {}
            done_dict = {}
            info_dict = {}

            done_dict["__all__"] = False
            for actor_id, action in action_dict.items():
                obs, reward, done, info = self._step(actor_id, action)
                obs_dict[actor_id] = obs
                reward_dict[actor_id] = reward
                done_dict[actor_id] = done
                if sum(list(done_dict.values())) == len(action_dict):
                    done_dict["__all__"] = True
                info_dict[actor_id] = info
            return obs_dict, reward_dict, done_dict, info_dict
        except Exception as xception:
            print("Error during step, terminating episode early",
                  xception)  # traceback.format_exc())

            self.clear_server_state()
            return self.last_obs, 0.0, True, {}

    def _step(self, actor_id, action):
        """Perform the actual step in the CARLA environment

        Applies control to `actor_id` based on `action`, process measurements,
        compute the rewards and terminal state info (dones).

        Args:
            actor_id(str): Actor identifier
            action: Actions to be executed for the actor.

        Returns
            obs (dict): Observations for each actor.
            rewards (dict): Reward values for each actor. None for first step
            dones (dict): Done values for each actor. Special key "__all__" is
                set when all actors are done and the env terminates
            info (dict): Info for each actor.
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

        self.last_reward[actor_id] = reward
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
            if not self.measurements_file:
                self.measurements_file = open(
                    os.path.join(
                        CARLA_OUT_PATH,
                        "measurements_{}.json".format(self.episode_id)), "w")
            self.measurements_file.write(json.dumps(py_measurements))
            self.measurements_file.write("\n")
            if done:
                self.measurements_file.close()
                self.measurements_file = None
                # if self.config["convert_images_to_video"] and (not self.video)
                #    self.images_to_video()
                #    self.video = Trueseg_city_space

        original_image = self.camera_list.cam_list[actor_id].image
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
            next_command = COMMANDS_ENUM[self.planner.get_next_command(
                [cur.get_location().x,
                 cur.get_location().y, GROUND_Z], [
                     cur.get_transform().rotation.pitch,
                     cur.get_transform().rotation.yaw, GROUND_Z
                 ], [
                     self.end_pos[actor_id][0], self.end_pos[actor_id][1],
                     GROUND_Z
                 ], [0.0, 90.0, GROUND_Z])]
        else:
            next_command = "LANE_FOLLOW"

        collision_vehicles = self.collisions[actor_id].collision_vehicles
        collision_pedestrians = self.collisions[actor_id].collision_pedestrians
        collision_other = self.collisions[actor_id].collision_other
        intersection_otherlane = self.lane_invasions[actor_id].offlane
        intersection_offroad = self.lane_invasions[actor_id].offroad

        self.previous_actions[actor_id] = next_command
        self.previous_rewards[actor_id] = self.last_reward[actor_id]

        if next_command == "REACH_GOAL":
            distance_to_goal = 0.0  # avoids crash in planner
        elif planner_enabled:
            distance_to_goal = self.planner.get_shortest_path_distance(
                [cur.get_location().x,
                 cur.get_location().y, GROUND_Z], [
                     cur.get_transform().rotation.pitch,
                     cur.get_transform().rotation.yaw, GROUND_Z
                 ], [
                     self.end_pos[actor_id][0], self.end_pos[actor_id][1],
                     GROUND_Z
                 ], [0, 90, 0]) / 100
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
            "episode_id": self.episode_id,
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
            "previous_actions": {
                actor_id: self.previous_actions[actor_id]
            },
            "previous_rewards": {
                actor_id: self.previous_rewards[actor_id]
            },
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
    m = py_measurements
    collided = (m["collision_vehicles"] > 0 or m["collision_pedestrians"] > 0
                or m["collision_other"] > 0)
    return bool(collided or m["total_reward"] < -100)


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

    for _ in range(1):
        multi_env_config = json.load(open(args.config))

        env = MultiCarlaEnv(multi_env_config)
        obs = env.reset()
        total_vehicle = env.num_vehicle

        total_reward_dict = {}
        action_dict = {}

        tmp = iter(multi_env_config.values())
        env_config = next(tmp)
        actor_configs = next(tmp)
        for actor_id in actor_configs.keys():
            total_reward_dict[actor_id] = 0

            #  Initialize all vehicles' action to be 3
            if env.discrete_actions:
                action_dict[actor_id] = 3
            else:
                action_dict[actor_id] = [1, 0]  # test number

        # server_clock = pygame.time.Clock()
        # print(server_clock.get_fps())

        start = time.time()
        i = 0
        # while not done["__all__"]:
        while i < 10:  # TEST
            i += 1
            obs, reward, done, info = env.step(action_dict)
            action_dict = get_next_actions(info, env.discrete_actions)
            for actor_id in total_reward_dict.keys():
                total_reward_dict[actor_id] += reward[actor_id]
            print(":{}\n\t".join(["Step#", "rew", "ep_rew", "done{}"]).format(
                i, reward, total_reward_dict, done))

            # Set done[__all__] for env termination when all agents are done
            done_temp = True
            for d in done:
                done_temp = done_temp and done[d]
            done["__all__"] = done_temp
            time.sleep(0.1)

        print("{} fps".format(i / (time.time() - start)))

        # Clean actors in world
        env.clean_world()

        # env.camera_list.save_images_to_disk()
