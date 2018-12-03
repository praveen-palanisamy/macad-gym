"""
multi_env.py: Multi-actor environment interface for CARLA-Gym
Should support two modes of operation. See CARLA-Gym developer guide for more info
__author__: PP, BP
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from enum import Enum

from datetime import datetime
import sys
import os
import glob

sys.path.append(
    glob.glob(f'**/**/PythonAPI/lib/carla-*{sys.version_info.major}.'
              f'{sys.version_info.minor}-linux-x86_64.egg')[0])

from env.multi_actor_env import *
from env.carla.PythonAPI.manual_control import HUD, CameraManager
import argparse  #pygame
import logging  #pygame
try:
    import pygame
    from pygame.locals import K_DOWN
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SPACE
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError(
        'cannot import pygame, make sure pygame package is installed')

import atexit
import cv2

import json
import random
import signal
import subprocess
import sys
import time
import traceback
import GPUtil
import carla
import numpy as np
import collections

import weakref  # for collision
import math  # for collision

try:
    import scipy.misc
except Exception:
    pass

import gym
from gym.spaces import Box, Discrete, Tuple
from .scenarios import *

#from .carla.settings import CarlaSettings
from env.carla.carla.planner import *

# Set this where you want to save image outputs (or empty string to disable)
CARLA_OUT_PATH = os.environ.get("CARLA_OUT", os.path.expanduser("~/carla_out"))
if CARLA_OUT_PATH and not os.path.exists(CARLA_OUT_PATH):
    os.makedirs(CARLA_OUT_PATH)

# Set this to the path of your Carla binary
SERVER_BINARY = os.environ.get(
    "CARLA_SERVER", os.path.expanduser("~/software/CARLA_0.9.1/CarlaUE4.sh"))

assert os.path.exists(SERVER_BINARY)

#  Assign initial value since they are not importable from an old APT carla.planner
REACH_GOAL = ""
GO_STRAIGHT = ""
TURN_RIGHT = ""
TURN_LEFT = ""
LANE_FOLLOW = ""
POS_COOR_MAP = None
# Number of vehicles/cars
#NUM_VEHICLE = 1

# Number of max step
MAX_STEP = 1000

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

# The cam for pygame
GLOBAL_CAM_POS = carla.Transform(carla.Location(x=178, y=198, z=40))

live_carla_processes = set()


def cleanup():
    print("Killing live carla processes", live_carla_processes)
    for pgid in live_carla_processes:
        os.killpg(pgid, signal.SIGKILL)


def termination_cleanup(*args):
    cleanup()
    sys.exit(0)


signal.signal(signal.SIGTERM, termination_cleanup)
signal.signal(signal.SIGINT, termination_cleanup)
atexit.register(cleanup)


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._history = []
        self._parent = parent_actor
        self._hud = hud
        self.offlane = 0
        self.offroad = 0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_detector')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    def get_invasion_history(self):
        history = collections.defaultdict(int)
        for frame, text in self._history:
            history[frame] = text
        return history

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return


#        text = ['%r' % str(x).split()[-1] for x in set(event.crossed_lane_markings)]
#        self._hud.notification('Crossed line %s' % ' and '.join(text))
        text = [
            '%r' % str(x).split()[-1] for x in set(event.crossed_lane_markings)
        ]
        self.offlane += 1
        print('VEHICLE %s' % (self._parent).id +
              ' crossed line %s' % ' and '.join(text))

        #  if one means not cross two lanes, means cross to offroad
        if len(set(event.crossed_lane_markings)) == 1:
            self.offroad += 1
            print('VEHICLE %s' % (self._parent).id +
                  ' crossed road %s' % ' and '.join(text))

        self._history.append((event.frame_number, text))
        if len(self._history) > 4000:
            self._history.pop(0)


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._history = []
        self._parent = parent_actor
        self._hud = hud
        self.collision_vehicles = 0
        self.collision_pedestrains = 0
        self.collision_other = 0
        self.collision_id_set = set()
        self.collision_type_id_set = set()
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self._history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return


#        actor_type = get_actor_display_name(event.other_actor)
#        self._hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self._history.append((event.frame_number, intensity))
        if len(self._history) > 4000:
            self._history.pop(0)
        print('vehicle %s ' % (self._parent).id +
              ' collision with %2d vehicles, %2d people, %2d others' %
              self.dynamic_collided())
        _cur = event.other_actor
        if _cur.id == 0:  #the static world objects
            if _cur.type_id in self.collision_type_id_set:
                return
            else:
                self.collision_type_id_set.add(_cur.type_id)
        else:
            if _cur.id in self.collision_id_set:
                return
            else:
                self.collision_id_set.add(_cur.id)

        collided_type = type(_cur).__name__
        if collided_type == 'Vehicle':
            self.collision_vehicles += 1
        elif collided_type == 'Pedestrain':
            self.collision_pedestrains += 1
        elif collided_type == 'Actor':
            self.collision_other += 1
        else:
            pass

    def _reset(self):
        self.collision_vehicles = 0
        self.collision_pedestrains = 0
        self.collision_other = 0
        self.collision_id_set = set()
        self.collision_type_id_set = set()

    def dynamic_collided(self):
        return (self.collision_vehicles, self.collision_pedestrains,
                self.collision_other)


class MultiCarlaEnv(MultiActorEnv):  #MultiActorEnv
    def __init__(self, args):  #config=ENV_CONFIG

        config_name = args.config

        #config=ENV_CONFIG
        self.config_list = json.load(open(config_name))

        # Get general/same config for actors.
        # For now , it contains map, city, discrete_actions, image_space, observation_space.
        general_config = self.config_list["0"]
        self.server_map = general_config["server_map"]
        self.city = self.server_map.split("/")[-1]
        self.render = general_config["render"]
        self.framestack = general_config["framestack"]
        self.discrete_actions = general_config["discrete_actions"]
        self.squash_action_logits = general_config["squash_action_logits"]
        self.verbose = general_config["verbose"]
        self.render_x_res = general_config["render_x_res"]
        self.render_y_res = general_config["render_y_res"]
        self.x_res = general_config["x_res"]
        self.y_res = general_config["y_res"]

        self.planner_list = []

        # For arg map.
        #self.config["server_map"] = "/Game/Carla/Maps/" + args.map

        for k in self.config_list:
            config = self.config_list[k]
            #config["scenarios"] = self.get_scenarios(args.scenario, config)
            if config["enable_planner"]:
                self.planner_list.append(Planner(self.city))
        #print(self.config_list)

        if self.city == "Town01":
            self.pos_coor_map = json.load(
                open("env/carla/POS_COOR/pos_cordi_map_town1.txt"))
        else:
            self.pos_coor_map = json.load(
                open("env/carla/POS_COOR/pos_cordi_map_town2.txt"))
        print(self)

        # TODO(ekl) this isn't really a proper gym spec
        self._spec = lambda: None
        self._spec.id = "Carla-v0"

        self.server_port = None
        self.server_process = None
        self.client = None
        self.num_steps = [0]
        self.total_reward = [0]
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
        self.image_pool = []
        self.previous_actions = []
        self.previous_rewards = []
        self.last_reward = []

    def get_scenarios(self, choice):

        if choice == "DEFAULT_SCENARIO_TOWN1":
            #config["server_map"] = "/Game/Carla/Maps/Town01"
            return DEFAULT_SCENARIO_TOWN1
        elif choice == "DEFAULT_SCENARIO_TOWN1_2":
            return DEFAULT_SCENARIO_TOWN1_2
        elif choice == "DEFAULT_SCENARIO_TOWN2":
            #config["server_map"] = "/Game/Carla/Maps/Town02"
            return DEFAULT_SCENARIO_TOWN2
        elif choice == "TOWN1_STRAIGHT":
            #config["server_map"] = "/Game/Carla/Maps/Town01"
            return TOWN1_STRAIGHT
        elif choice == "CURVE_TOWN1":
            #config["server_map"] = "/Game/Carla/Maps/Town01"
            return CURVE_TOWN1
        elif choice == "CURVE_TOWN2":
            #config["server_map"] = "/Game/Carla/Maps/Town02"
            return CURVE_TOWN2
        elif choice == "DEFAULT_CURVE_TOWN1":
            #config["server_map"] = "/Game/Carla/Maps/Town01"
            return DEFAULT_CURVE_TOWN1

    def init_server(self):

        print("Initializing new Carla server...")
        # Create a new server process and start the client.
        self.server_port = 2000
        gpus = GPUtil.getGPUs()
        print('Get gpu:')
        if not self.render and (gpus is not None and len(gpus)) > 0:
            min_index = random.randint(0, len(gpus) - 1)
            for i, gpu in enumerate(gpus):
                if gpu.load < gpus[min_index].load:
                    min_index = i

            self.server_process = subprocess.Popen(
                ("DISPLAY=:8 vglrun -d :7.{} {} " + self.server_map +
                 " -windowed -ResX=800 -ResY=600 -carla-server -carla-world-port={}"
                 ).format(min_index, SERVER_BINARY, self.server_port),
                shell=True,
                preexec_fn=os.setsid,
                stdout=open(os.devnull, "w"))
        else:
            self.server_process = subprocess.Popen(
                [
                    SERVER_BINARY,
                    self.server_map,
                    "-windowed",
                    "-ResX=800",
                    "-ResY=600",
                    #"-carla-settings=/home/fastisu/Documents/CARLA-Gym/bo_CS.ini",
                    "-benchmark -fps=10"
                    "-carla-server",
                    "-carla-world-port={}".format(self.server_port)
                ],
                preexec_fn=os.setsid,
                stdout=open(os.devnull, "w"))
        live_carla_processes.add(os.getpgid(self.server_process.pid))

        # wait for carlar server to start
        time.sleep(10)

        self.actor_list = []
        self.cam_list = []
        self.colli_list = []
        self.lane_list = []
        self.client = carla.Client("localhost", 2000)  #self.server_port)

    def clear_server_state(self):
        print("Clearing Carla server state")
        try:
            if self.client:
                self.client.disconnect()
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

    def __del__(self):
        self.clear_server_state()

    def reset(self):
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

    #  funcs _parse_image and _on_render are from manual_control.py,
    #     combine some command from _parse_image to preprocess_image().
    def _parse_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self._surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    def _on_render(self):
        if self._surface is not None:
            self._display.blit(self._surface, (0, 0))
        pygame.display.flip()

    def _reset(self):

        # If config contains a single scenario, then use it if it's an array of scenarios, randomly choose one and init
        self.scenario_list = []

        for k in self.config_list:
            config = self.config_list[k]
            scenario = self.get_scenarios(config["scenarios"])
            if isinstance(scenario, dict):
                self.scenario_list.append(scenario)
            else:  #ininstance array of dict
                self.scenario_list.append(random.choice(scenario))

        NUM_VEHICLE = len(self.scenario_list)
        self.num_vehicle = NUM_VEHICLE

        self.num_steps = [0] * self.num_vehicle
        self.total_reward = [0] * self.num_vehicle
        self.prev_measurement = None
        self.prev_image = None
        self.episode_id = datetime.today().strftime("%Y-%m-%d_%H-%M-%S_%f")
        self.measurements_file = None

        #  Create new camera in Carla_0.9.0.
        world = self.client.get_world()
        self.cur_map = world.get_map()
        #self.cur_map.save_to_disk(CARLA_OUT_PATH)
        #print(self.cur_map.get_spawn_points())

        #  Asynchronously camera test:
        #print('image000:', self.image)
        #time.sleep(5)
        #print('image111:', self.image)
        #time.sleep(5)
        #print('image222:', self.image)

        for n in range(self.num_vehicle):
            self.image_pool.append([])
            self.previous_actions.append([])
            self.previous_rewards.append([])

        POS_S = [[0] * 3] * self.num_vehicle
        POS_E = [[0] * 3] * self.num_vehicle

        self.last_reward = [0] * self.num_vehicle
        for i in range(self.num_vehicle):
            scenario = self.scenario_list[i]
            print(type(scenario))
            s_id = str(
                scenario["start_pos_id"]
            )  #str(start_id).decode("utf-8") # unicode is needed. this trans is for py2
            e_id = str(scenario["end_pos_id"])
            POS_S[i] = self.pos_coor_map[s_id]
            POS_E[i] = self.pos_coor_map[e_id]

        world = self.client.get_world()
        self.weather = [
            world.get_weather().cloudyness,
            world.get_weather().precipitation,
            world.get_weather().precipitation_deposits,
            world.get_weather().wind_intensity
        ]

        for i in range(self.num_vehicle):
            blueprints = world.get_blueprint_library().filter('vehicle')
            blueprint = random.choice(blueprints)
            #color = random.choice(blueprint.get_attribute('color').recommended_values)
            #blueprint.set_attribute('color', color)
            transform = carla.Transform(
                carla.Location(x=POS_S[i][0], y=POS_S[i][1], z=POS_S[i][2]),
                carla.Rotation(pitch=0, yaw=0, roll=0))
            print('spawning vehicle %r with %d wheels' %
                  (blueprint.id, blueprint.get_attribute('number_of_wheels')))
            vehicle = world.try_spawn_actor(blueprint, transform)
            print('vehicle at ',
                  vehicle.get_location().x,
                  vehicle.get_location().y,
                  vehicle.get_location().z)
            #time.sleep(1000)
            self.actor_list.append(vehicle)
            collision_sensor = CollisionSensor(vehicle, 0)
            lane_sensor = LaneInvasionSensor(vehicle, 0)
            self.colli_list.append(collision_sensor)
            self.lane_list.append(lane_sensor)

            if i == 0:  #TEST!!
                config = self.config_list[str(i)]
                cam_type, cc = self.get_camera(config)
                self.cc = cc
                cam_blueprint = world.get_blueprint_library().find(cam_type)
                camera = world.spawn_actor(
                    cam_blueprint,
                    carla.Transform(
                        carla.Location(x=-5.5, z=2.8),
                        carla.Rotation(pitch=-15)),
                    attach_to=self.actor_list[i])
                self.cam_list.append(camera)
                self.cam_list[i].listen(
                    lambda image: self.get_image(image, i, cc))

                #wait the camera's launching time to get first image
                print("camera finished")
                time.sleep(3)
        self.cam_start = time.time()
        print('All vehicles are created.')

        #UNDER TEST START-----------------------------------------------------------------------------------
        debugger = world.debug
        location = carla.Location(
            x=178.7699737548828, y=198.75999450683594, z=39.430625915527344)
        debugger.draw_point(
            location,
            size=0.1,
            color=carla.Color(),
            life_time=-1.0,
            persistent_lines=True)
        print(location.distance)
        print(self.cur_map.get_waypoint(location, project_to_road=True))
        print(
            self.cur_map.get_waypoint(location,
                                      project_to_road=True).lane_width)
        print(type(self.cur_map.get_waypoint(location, project_to_road=True)))
        for wp in self.cur_map.generate_waypoints(1.0):
            debugger.draw_point(
                wp.transform.location,
                size=0.1,
                color=carla.Color(),
                life_time=-1.0,
                persistent_lines=True)
        print(type(self.cur_map.generate_waypoints(1.0)))
        #UNDER TEST END-------------------------------------------------------------------------------------

        #  Need to print for multiple client
        self.start_pos = POS_S
        self.end_pos = POS_E
        self.start_coord = []
        self.end_coord = []
        self.py_measurement = {}
        self.prev_measurement = {}
        self.obs = []

        for i in range(self.num_vehicle):
            self.start_coord.append(
                [self.start_pos[i][0] // 100, self.start_pos[i][1] // 100])
            self.end_coord.append(
                [self.end_pos[i][0] // 100, self.end_pos[i][1] // 100])

            print("Client {} start pos {} ({}), end {} ({})".format(
                i, self.start_pos[i], self.start_coord[i], self.end_pos[i],
                self.end_coord[i]))

            # Notify the server that we want to start the episode at the
            # player_start index. This function blocks until the server is ready
            # to start the episode.

            #  no episode block in 0.9.0
            #print("Starting new episode...")
            #self.client.start_episode(self.scenario["start_pos_id"])

            #  start read observation. each loop read one vehcile infor
            py_mt = self._read_observation(i)
            vehcile_name = 'Vehcile'
            vehcile_name += str(i)
            self.py_measurement[vehcile_name] = py_mt
            self.prev_measurement[vehcile_name] = py_mt

            obs = self.encode_obs(self.image,
                                  self.py_measurement[vehcile_name], i)
            self.obs_dict[vehcile_name] = obs

        return self.obs_dict

    def get_camera(self, config):
        if config["camera_type"] == "rgb":
            camera_type = 'sensor.camera.rgb'
            cc = carla.ColorConverter.Raw
            config["use_depth_camera"] = False
        elif config["camera_type"] == "depth_raw":
            camera_type = 'sensor.camera.depth'
            cc = carla.ColorConverter.Raw
            config["use_depth_camera"] = False
        elif config["camera_type"] == "depth_gray_scale":
            camera_type = 'sensor.camera.depth'
            cc = carla.ColorConverter.Depth
            config["use_depth_camera"] = True
        elif config["camera_type"] == "depth_log_gray_scale":
            camera_type = 'sensor.camera.depth'
            cc = carla.ColorConverter.LogarithmicDepth
            config["use_depth_camera"] = True
        elif config["camera_type"] == "seg_raw":
            camera_type = 'sensor.camera.semantic_segmentation'
            cc = carla.ColorConverter.Raw
            config["use_depth_camera"] = False
        elif config["camera_type"] == "seg_city_space":
            camera_type = 'sensor.camera.semantic_segmentation'
            cc = carla.ColorConverter.CityScapesPalette
            config["use_depth_camera"] = False
        return camera_type, cc

    def get_image(self, image, i, cc):

        #image_dir = os.path.join(CARLA_OUT_PATH, 'images/{}/{}_%04d.png'.format(i,self.episode_id) % image.frame_number)
        #image.save_to_disk(image_dir, cc)
        self.image_pool[i].append(image)
        self.original_image = image
        self._parse_image(image)  # py_game render use
        self.image = self.preprocess_image(image, i)

    def encode_obs(self, image, py_measurements, vehcile_number):

        assert self.framestack in [1, 2]
        # currently, the image is generated asynchronously
        prev_image = self.prev_image
        self.prev_image = image
        if prev_image is None:
            prev_image = image
        if self.framestack == 2:
            #image = np.concatenate([prev_image, image], axis=2)
            image = np.concatenate([prev_image, image])
        obs = ('Vehcile number: ', vehcile_number, image,
               COMMAND_ORDINAL[py_measurements["next_command"]], [
                   py_measurements["forward_speed"],
                   py_measurements["distance_to_goal"]
               ])
        self.last_obs = obs
        return obs

    def step(self, action_dict):
        try:
            obs_dict = {}
            reward_dict = {}
            done_dict = {}
            info_dict = {}

            actor_num = 0
            for action in action_dict:
                obs, reward, done, info = self._step(action_dict[action],
                                                     actor_num)

                vehcile_name = 'Vehcile'
                vehcile_name += str(actor_num)
                actor_num += 1
                obs_dict[vehcile_name] = obs
                reward_dict[vehcile_name] = reward
                done_dict[vehcile_name] = done
                info_dict[vehcile_name] = info
            #self.step_number += 1
            return obs_dict, reward_dict, done_dict, info_dict
        except Exception:
            print("Error during step, terminating episode early",
                  traceback.format_exc())
            self.clear_server_state()
            return (self.last_obs, 0.0, True, {})

    def _step(self, action, i):
        if self.discrete_actions:
            action = DISCRETE_ACTIONS[int(action)]
        assert len(action) == 2, "Invalid action {}".format(action)
        if self.squash_action_logits:
            forward = 2 * float(sigmoid(action[0]) - 0.5)
            throttle = float(np.clip(forward, 0, 1))
            brake = float(np.abs(np.clip(forward, -1, 0)))
            steer = 2 * float(sigmoid(action[1]) - 0.5)
        else:
            throttle = float(np.clip(action[0], 0, 0.6))  #10
            brake = float(np.abs(np.clip(action[0], -1, 0)))
            steer = float(np.clip(action[1], -1, 1))
        reverse = False
        hand_brake = False
        if self.verbose:
            print("steer", steer, "throttle", throttle, "brake", brake,
                  "reverse", reverse)

        #  send control
        config = self.config_list[str(i)]
        if config['manual_control']:
            if i == 0:
                #pygame need this
                self._display = pygame.display.set_mode(
                    (800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
                logging.debug('pygame started')

                control1 = self._get_keyboard_control1(
                    pygame.key.get_pressed())
                self.actor_list[i].apply_control(control1)
                self._on_render()
            else:
                self._display = pygame.display.set_mode(
                    (800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
                logging.debug('pygame started')

                control2 = self._get_keyboard_control2(
                    pygame.key.get_pressed())
                self.actor_list[i].apply_control(control2)
                self._on_render()
        elif config["auto_control"]:
            self.actor_list[i].set_autopilot()
        else:
            # Test waypoints
            #next_point_transform = self.get_transform_from_nearest_way_point(i, self.end_pos)
            #next_point_transform.location.z = 40 # the point with z = 0, and the default z of cars are 40
            #self.actor_list[i].set_transform(next_point_transform)
            self.actor_list[i].apply_control(
                carla.VehicleControl(
                    throttle=throttle,
                    steer=steer,
                    brake=brake,
                    hand_brake=hand_brake,
                    reverse=reverse))

        # Process observations
        py_measurements = self._read_observation(i)
        print('<<<<<<', py_measurements["collision_other"])
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
        vehcile_name = 'Vehcile'
        vehcile_name += str(i)
        reward = compute_reward(self, self.prev_measurement[vehcile_name],
                                py_measurements, i)
        self.last_reward[
            i] = reward  # to make the previous_rewards in py_measurements
        #  update num_steps and total_reward lists if next car comes
        #if i == len(self.num_steps):
        #    self.num_steps.append(0)
        #if i == len(self.total_reward):
        #    self.total_reward.append(0)

        self.total_reward[i] += reward
        py_measurements["reward"] = reward
        py_measurements["total_reward"] = self.total_reward
        done = (
            self.num_steps[i] > MAX_STEP or  #self.scenario["max_steps"] or
            py_measurements["next_command"] == "REACH_GOAL")  # or
        #(self.config["early_terminate_on_collision"] and
        # collided_done(py_measurements)))
        py_measurements["done"] = done

        self.prev_measurement[vehcile_name] = py_measurements
        self.num_steps[i] += 1
        print('>>>', py_measurements["collision_other"])
        # Write out measurements to file
        if i == self.num_vehicle - 1:  #print all cars measurement
            if CARLA_OUT_PATH:
                if not self.measurements_file:
                    self.measurements_file = open(
                        os.path.join(
                            CARLA_OUT_PATH, "measurements_{}.json".format(
                                self.episode_id)), "w")
                self.measurements_file.write(json.dumps(py_measurements))
                self.measurements_file.write("\n")
                if done:
                    self.measurements_file.close()
                    self.measurements_file = None
                    #if self.config["convert_images_to_video"] and (not self.video):
                    #    self.images_to_video()
                    #    self.video = Trueseg_city_space
        image = self.preprocess_image(self.original_image, i)
        return (self.encode_obs(image, py_measurements, i), reward, done,
                py_measurements)

    def get_transform_from_nearest_way_point(self, vehicle_id, pos):
        vehcile = self.actor_list[vehicle_id]
        way_points = self.cur_map.get_waypoint(vehcile.get_location())
        nexts = list(way_points.next(1.0))
        print('Next(1.0) --> %d waypoints' % len(nexts))
        if not nexts:
            raise RuntimeError("No more waypoints!")
        smallest_dist = sys.maxsize
        for p in nexts:
            trans = p.transform.location
            diff_x = trans.x - pos[vehicle_id][0]
            diff_y = trans.y - pos[vehicle_id][1]
            diff_z = trans.z - pos[vehicle_id][2]
            cur_dist = np.linalg.norm([diff_x, diff_y, diff_z])
            if cur_dist < smallest_dist:
                next_point = p
        text = "road id = %d, lane id = %d, transform = %s"
        print(text % (next_point.road_id, next_point.lane_id,
                      next_point.transform))

        #debugger = self.client.get_world().debug
        #debugger.draw_point(next_point.transform.location, size=0.1, color=carla.Color(), life_time=-1.0, persistent_lines=True)
        return next_point.transform

    def _get_keyboard_control1(self, keys):
        control = carla.VehicleControl()
        if keys[K_a]:
            control.steer = -1.0
        if keys[K_d]:
            control.steer = 1.0
        if keys[K_w]:
            control.throttle = 1.0
        if keys[K_s]:
            control.brake = 1.0
        if keys[K_SPACE]:
            control.hand_brake = True
        if keys[K_q]:
            control.reverse = not control.reverse
        #if keys[K_p]:
        #    self._autopilot_enabled = not self._autopilot_enabled
        #control.reverse = self._is_on_reverse
        return control

    def _get_keyboard_control2(self, keys):
        control = carla.VehicleControl()
        if keys[K_LEFT]:
            control.steer = -1.0
        if keys[K_RIGHT]:
            control.steer = 1.0
        if keys[K_UP]:
            control.throttle = 1.0
        if keys[K_DOWN]:
            control.brake = 1.0
        if keys[K_SPACE]:
            control.hand_brake = True
        if keys[K_q]:
            control.reverse = not control.reverse
        #if keys[K_p]:
        #    self._autopilot_enabled = not self._autopilot_enabled
        #control.reverse = self._is_on_reverse
        return control

    def images_to_video(self):
        videos_dir = os.path.join(CARLA_OUT_PATH, "Videos")
        if not os.path.exists(videos_dir):
            os.makedirs(videos_dir)

        ffmpeg_cmd = (
            "ffmpeg -loglevel -8 -r 20 -f image2 -s {x_res}x{y_res} "
            "-pattern_type glob "
            "-i '{img}/*.png' -vcodec libx264 {vid}.mp4 && rm -f {img}/*.png"  #&& rm -f {img}/*.png
        ).format(
            x_res=self.render_x_res,
            y_res=self.render_y_res,
            #first_frame_num = self.first_frame_num,
            vid=os.path.join(videos_dir, self.episode_id),
            img=os.path.join(CARLA_OUT_PATH, "images"))
        print("Executing ffmpeg command", ffmpeg_cmd)
        subprocess.call(ffmpeg_cmd, shell=True)

    def preprocess_image(self, image, i):
        config = self.config_list[str(i)]
        if config["use_depth_camera"]:
            assert config["use_depth_camera"]
            data = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            data = np.reshape(data, (self.render_y_res, self.render_x_res, 4))
            data = data[:, :, :1]
            data = data[:, :, ::-1]
            data = cv2.resize(
                data, (self.x_res, self.y_res), interpolation=cv2.INTER_AREA)
            data = np.expand_dims(data, 2)
        else:
            data = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            data = np.reshape(data, (self.render_y_res, self.render_x_res, 4))
            data = data[:, :, :3]
            data = data[:, :, ::-1]
            data = cv2.resize(
                data, (self.x_res, self.y_res), interpolation=cv2.INTER_AREA)
            data = (data.astype(np.float32) - 128) / 128

        return data

    def _read_observation(self, i):
        # Read the data produced by the server this frame.
        #  read_data() depends tcp from old API. carla/PythonClient/carla/client.py
        #measurements, sensor_data = self.client.read_data()

        # Print some of the measurements.
        #  set verbose false, because donot know what measurements from read_data is.
        #if self.config["verbose"]:
        #    print_measurements(measurements)

        #  Old API of cameras.
        #observation = None
        #if self.config["use_depth_camera"]:
        #    camera_name = "CameraDepth"
        #else:
        #    camera_name = "CameraRGB"
        #for name, image in sensor_data.items():
        #if name == camera_name:
        #observation = image

        print(type(self.actor_list[i].get_transform().rotation.pitch))
        print(type(0.0))
        #time.sleep(1000)
        #cur = measurements.player_measurements
        cur = self.actor_list[i]
        cur_config = self.config_list[str(i)]
        planner_enabled = cur_config["enable_planner"]
        if planner_enabled:
            next_command = COMMANDS_ENUM[self.planner_list[i].get_next_command(
                [cur.get_location().x,
                 cur.get_location().y, GROUND_Z], [
                     cur.get_transform().rotation.pitch,
                     cur.get_transform().rotation.yaw, GROUND_Z
                 ], [self.end_pos[i][0], self.end_pos[i][1], GROUND_Z],
                [0.0, 90.0, GROUND_Z])]
        else:
            next_command = "LANE_FOLLOW"
        #time.sleep(1000)
        collision_vehicles = self.colli_list[i].collision_vehicles
        collision_pedestrians = self.colli_list[i].collision_pedestrains
        collision_other = self.colli_list[i].collision_other
        intersection_otherlane = self.lane_list[i].offlane
        intersection_offroad = self.lane_list[i].offroad
        print("---->", collision_other)

        #  A simple planner
        #current_x = self.actor_list[i].get_location().x
        #current_y = self.actor_list[i].get_location().y

        #distance_to_goal_euclidean = float(np.linalg.norm(
        #    [current_x - self.end_pos[i][0],
        #     current_y - self.end_pos[i][1]]) / 100)

        #distance_to_goal = distance_to_goal_euclidean

        #diff_x =  abs(current_x - self.end_pos[i][0])
        #diff_y =  abs(current_y - self.end_pos[i][1])

        #diff_s_x = abs(current_x - self.start_pos[i][0])
        #diff_s_y = abs(current_y - self.start_pos[i][1])

        #next_command = "LANE_FOLLOW"
        self.previous_actions[i].append(next_command)

        #if diff_x < 1 and diff_y < 1:
        #if current_x - self.end_pos[i][0] > 0:
        #if (diff_s_x + diff_s_y > 30) or (diff_x < 5 and diff_y < 5):  # ONLY FOR TEST
        #    next_command = "REACH_GOAL"

        self.previous_rewards[i].append(self.last_reward[i])

        if next_command == "REACH_GOAL":
            distance_to_goal = 0.0  # avoids crash in planner
        elif planner_enabled:
            distance_to_goal = self.planner_list[i].get_shortest_path_distance(
                [cur.get_location().x,
                 cur.get_location().y, GROUND_Z], [
                     cur.get_transform().rotation.pitch,
                     cur.get_transform().rotation.yaw, GROUND_Z
                 ], [self.end_pos[i][0], self.end_pos[i][1], GROUND_Z],
                [0, 90, 0]) / 100
        else:
            distance_to_goal = -1

        distance_to_goal_euclidean = float(
            np.linalg.norm([
                self.actor_list[i].get_location().x - self.end_pos[i][0],
                self.actor_list[i].get_location().y - self.end_pos[i][1]
            ]) / 100)
        py_measurements = {
            "episode_id": self.episode_id,
            "step": self.num_steps[i],
            "x": self.actor_list[i].get_location().x,
            "y": self.actor_list[i].get_location().y,
            "pitch": self.actor_list[i].get_transform().rotation.pitch,
            "yaw": self.actor_list[i].get_transform().rotation.yaw,
            "roll": self.actor_list[i].get_transform().rotation.roll,
            "forward_speed": self.actor_list[i].get_velocity().x,
            "distance_to_goal": distance_to_goal,  #use planner
            "distance_to_goal_euclidean": distance_to_goal_euclidean,
            "collision_vehicles": collision_vehicles,
            "collision_pedestrians": collision_pedestrians,
            "collision_other": collision_other,
            "intersection_offroad": intersection_offroad,
            "intersection_otherlane": intersection_otherlane,
            "weather": self.weather,
            "map": self.server_map,
            "start_coord": self.start_coord[i],
            "end_coord": self.end_coord[i],
            "current_scenario": self.scenario_list[i],
            "x_res": self.x_res,
            "y_res": self.y_res,
            "num_vehicles": self.scenario_list[i]["num_vehicles"],
            "num_pedestrians": self.scenario_list[i]["num_pedestrians"],
            "max_steps": 1000,  # set 1000 now. self.scenario["max_steps"],
            "next_command": next_command,
            "previous_actions": {
                i: self.previous_actions[i]
            },
            "previous_rewards": {
                i: self.previous_rewards[i]
            }
        }

        #  self.original_image.save_to_disk can also implemented here.
        #save_to_disk(self.original_image)
        #if CARLA_OUT_PATH and self.config["log_images"]:
        #    for name, image in sensor_data.items():
        #        out_dir = os.path.join(CARLA_OUT_PATH, name)
        #        if not os.path.exists(out_dir):
        #            os.makedirs(out_dir)
        #        out_file = os.path.join(
        #            out_dir,
        #            "{}_{:>04}.jpg".format(self.episode_id, self.num_steps))
        #        scipy.misc.imsave(out_file, image.data)

        #assert observation is not None, sensor_data

        return py_measurements


def compute_reward_corl2017(env, prev, current):
    reward = 0.0

    cur_dist = current["distance_to_goal"]

    prev_dist = prev["distance_to_goal"]

    if env.verbose:
        print("Cur dist {}, prev dist {}".format(cur_dist, prev_dist))

    # Distance travelled toward the goal in m
    reward += np.clip(prev_dist - cur_dist, -10.0, 10.0)

    # Change in speed (km/h)
    reward += 0.05 * (current["forward_speed"] - prev["forward_speed"])

    #  no collision and sidewarlk now.
    # New collision damage
    reward -= .00002 * (
        current["collision_vehicles"] + current["collision_pedestrians"] +
        current["collision_other"] - prev["collision_vehicles"] -
        prev["collision_pedestrians"] - prev["collision_other"])

    # New sidewalk intersection
    reward -= 2 * (
        current["intersection_offroad"] - prev["intersection_offroad"])

    # New opposite lane intersection
    reward -= 2 * (
        current["intersection_otherlane"] - prev["intersection_otherlane"])

    return reward


def compute_reward_custom(env, prev, current):
    reward = 0.0

    cur_dist = current["distance_to_goal"]
    prev_dist = prev["distance_to_goal"]

    if env.verbose:
        print("Cur dist {}, prev dist {}".format(cur_dist, prev_dist))

    # Distance travelled toward the goal in m
    reward += np.clip(prev_dist - cur_dist, -10.0, 10.0)

    # Speed reward, up 30.0 (km/h)
    reward += np.clip(current["forward_speed"], 0.0, 30.0) / 10

    # New collision damage
    new_damage = (
        current["collision_vehicles"] + current["collision_pedestrians"] +
        current["collision_other"] - prev["collision_vehicles"] -
        prev["collision_pedestrians"] - prev["collision_other"])
    if new_damage:
        reward -= 100.0

    # Sidewalk intersection
    reward -= current["intersection_offroad"]

    # Opposite lane intersection
    reward -= current["intersection_otherlane"]

    # Reached goal
    if current["next_command"] == "REACH_GOAL":
        reward += 100.0

    return reward


def compute_reward_lane_keep(env, prev, current):
    reward = 0.0

    # Speed reward, up 30.0 (km/h)
    reward += np.clip(current["forward_speed"], 0.0, 30.0) / 10

    # New collision damage
    new_damage = (
        current["collision_vehicles"] + current["collision_pedestrians"] +
        current["collision_other"] - prev["collision_vehicles"] -
        prev["collision_pedestrians"] - prev["collision_other"])
    if new_damage:
        reward -= 100.0

    # Sidewalk intersection
    reward -= current["intersection_offroad"]

    # Opposite lane intersection
    reward -= current["intersection_otherlane"]

    return reward


REWARD_FUNCTIONS = {
    "corl2017": compute_reward_corl2017,
    "custom": compute_reward_custom,
    "lane_keep": compute_reward_lane_keep,
}


def compute_reward(env, prev, current, i):
    config = env.config_list[str(i)]
    return REWARD_FUNCTIONS[config["reward_function"]](env, prev, current)


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


def get_next_actions(measurements, action_dict):
    v = 0
    for k in measurements:
        m = measurements[k]
        command = m["next_command"]
        name = 'Vehcile' + str(v)
        if command == "REACH_GOAL":
            action_dict[name] = 0
        elif command == "GO_STRAIGHT":
            action_dict[name] = 3
        elif command == "TURN_RIGHT":
            action_dict[name] = 6
        elif command == "TURN_LEFT":
            action_dict[name] = 5
        elif command == "LANE_FOLLOW":
            action_dict[name] = 3
        v = v + 1
    return action_dict


if __name__ == "__main__":
    #  Episode for loop
    #from multi_env import MultiCarlaEnv
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
        #  Initialize server and clients.
        env = MultiCarlaEnv(args)
        print('env finished')
        obs = env.reset()
        print('obs infor:')
        print(obs)
        #time.sleep(1000) #  test use

        done = False
        i = 0
        total_vehcile = len(obs)
        #  Initialize total reward dict.
        total_reward_dict = {}
        for n in range(total_vehcile):
            vehcile_name = 'Vehcile'
            vehcile_name += str(n)
            total_reward_dict[vehcile_name] = 0

        #  Initialize all vehciles' action to be 3
        action_dict = {}
        for v in range(total_vehcile):
            vehcile_name = 'Vehcile' + str(v)
            action_dict[vehcile_name] = 3
        #  3 in action_list means go straight.
        #action_list = {
        #'Vehcile0' : 3,
        #'Vehcile1' : 3,
        #}
        #server_clock = pygame.time.Clock()
        #print(server_clock.get_fps())
        #time.sleep(1000)
        start = time.time()
        all_done = False
        #while not all_done:
        while i < 100:  # TEST
            i += 1
            if env.discrete_actions:
                obs, reward, done, info = env.step(action_dict)
                action_dict = get_next_actions(info, action_dict)
            else:
                obs, reward, done, info = env.step([0, 1, 0])

            for t in total_reward_dict:
                total_reward_dict[t] += reward[t]

            print("Step", i, "rew", reward, "total", total_reward_dict, "done",
                  done)

            #  Test whether all vehicles have finished.
            done_temp = True
            for d in done:
                done_temp = done_temp and done[d]
            all_done = done_temp
            time.sleep(0.1)
        print(obs)
        print(reward)
        print(done)
        total_time = time.time() - env.cam_start
        print("{} fps".format(i / (time.time() - start)))
        for cam in env.cam_list:
            cam.destroy()
        for actor in env.actor_list:
            actor.destroy()
        # Start save the images from image pool to disk:
        print("Saving the images from image pool to disk:")
        print("image frames:", len(env.image_pool[0]))
        print("cam time:", total_time)
        pool = env.image_pool[0]
        last_image = pool[-1]
        print("server fps:", (last_image.frame_number) / total_time)
        print("server fps:", len(env.image_pool[0]) / total_time)
        #print(len(env.image_pool[1]))

        for n in range(total_vehcile):
            for image in env.image_pool[n]:
                image_dir = os.path.join(
                    CARLA_OUT_PATH,
                    'images/{}/%04d.png'.format(n) % image.frame_number)
                image.save_to_disk(image_dir, env.cc)

        #env.images_to_video()
