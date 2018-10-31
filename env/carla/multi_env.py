"""OpenAI gym environment for Carla. Run this file for a demo."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from enum import Enum

from datetime import datetime
import sys

#sys.path.append(
#    'PythonAPI/carla-0.9.0-py%d.%d-linux-x86_64.egg' % (sys.version_info.major,
#                                                        sys.version_info.minor))
sys.path.append(
    'PythonAPI/carla-0.9.0-py3.6-linux-x86_64.egg')

import argparse#pygame
import logging#pygame
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
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')



import atexit
#import cv2
import os
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
try:
    import scipy.misc
except Exception:
    pass

import gym
from gym.spaces import Box, Discrete, Tuple
from scenarios import *
#import multi_actor_env
#from .multi_actor_env import MultiActorEnv
#from ..multi_actor_env import MultiActorEnv
#from scenarios import DEFAULT_SCENARIO_TOWN1,update_scenarios_parameter
from settings import CarlaSettings
# Set this where you want to save image outputs (or empty string to disable)
CARLA_OUT_PATH = os.environ.get("CARLA_OUT", os.path.expanduser("~/carla_out"))
if CARLA_OUT_PATH and not os.path.exists(CARLA_OUT_PATH):
    os.makedirs(CARLA_OUT_PATH)

# Set this to the path of your Carla binary
SERVER_BINARY = os.environ.get(
    "CARLA_SERVER", os.path.expanduser("~/software/CARLA_0.9.0/CarlaUE4.sh"))

assert os.path.exists(SERVER_BINARY)

#  Assign initial value since they are not importable from an old APT carla.planner
REACH_GOAL = ""
GO_STRAIGHT = ""
TURN_RIGHT = ""
TURN_LEFT = ""
LANE_FOLLOW = ""
POS_COOR_MAP = None 
# Number of vehicles/cars   
NUM_VEHICLE = 1

# Number of max step
MAX_STEP = 1000

# Set of the start and end position
#POS_S = []
#POS_E = []
#POS_S.append([180.0,199.0,40.0])
#POS_S.append([180.0,195.0,40.0])
#POS_E.append([200.0,199.0,40.0])
#POS_E.append([200.0,195.0,40.0])

# Carla planner commands
COMMANDS_ENUM = {
    REACH_GOAL : "REACH_GOAL",
    GO_STRAIGHT : "GO_STRAIGHT",
    TURN_RIGHT : "TURN_RIGHT",
    TURN_LEFT : "TURN_LEFT",
    LANE_FOLLOW : "LANE_FOLLOW",
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

# Default environment configuration
ENV_CONFIG = {
    "log_images": True,
    "enable_planner": False,
    "render": True,  # Whether to render to screen or send to VFB
    "framestack": 2,  # note: only [1, 2] currently supported
    "convert_images_to_video": False,
    "early_terminate_on_collision": True,
    "verbose": False,
    "reward_function": "corl2017",
    "render_x_res": 800,
    "render_y_res": 600,
    "x_res": 80,
    "y_res": 80,
    "server_map": "/Game/Carla/Maps/Town01", 
    "scenarios": {}, #[DEFAULT_SCENARIO_TOWN1], # no scenarios
    "use_depth_camera": False,
    "discrete_actions": True,
    "squash_action_logits": False,
    "manual_control": False,
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
    5: [1.0, -0.5],
    # forward right
    6: [1.0, 0.5],
    # brake left
    7: [-0.5, -0.5],
    # brake right
    8: [-0.5, 0.5],
}

# The cam for pygame
GLOBAL_CAM_POS = carla.Transform(carla.Location(x=170, y = 199, z = 45))

live_carla_processes = set()



def save_to_disk(image):
        """Save this image to disk (requires PIL installed)."""

        filename = '_images/{:0>6d}_{:s}.png'.format(image.frame_number, image.type)
    
        try:
            from PIL import Image as PImage
        except ImportError:
            raise RuntimeError(
                'cannot import PIL, make sure pillow package is installed')
        #image = PImage.frombytes()
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        image = PImage.frombytes(
            mode='RGBA',
            size=(image.width, image.height),
            data = array) # work in python3
            #data=image.raw_data, # work in python2
            #decoder_name='raw')  # work in python 2
        #color = image.split() # work in python2
        #image = PImage.merge("RGB", color[2::-1])
        
        folder = os.path.dirname(filename)
        if not os.path.isdir(folder):
            os.makedirs(folder)
        image.save(filename)   

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
 


class MultiCarlaEnv(): #MultiActorEnv
    def __init__(self, args):#config=ENV_CONFIG

        config_name = args.config
       
        #config=ENV_CONFIG
        config = json.load(open(config_name))
        print(config)
        
        self.config = config
        self.config["scenarios"] = self.get_scenarios(args.scenario)
        self.config["server_map"] = "/Game/Carla/Maps/" + args.map
        self.city = self.config["server_map"].split("/")[-1]
        if self.config["enable_planner"]:
            self.planner = Planner(self.city)

        if config["discrete_actions"]:
            self.action_space = Discrete(len(DISCRETE_ACTIONS))
        else:
            self.action_space = Box(-1.0, 1.0, shape=(2,))
        if config["use_depth_camera"]:
            image_space = Box(
                -1.0, 1.0, shape=(
                    config["y_res"], config["x_res"],
                    1 * config["framestack"]))
        else:
            image_space = Box(
                0.0, 255.0, shape=(
                    config["y_res"], config["x_res"],
                    3 * config["framestack"]))
        self.observation_space = Tuple(
            [image_space,
             Discrete(len(COMMANDS_ENUM)),  # next_command
             Box(-128.0, 128.0, shape=(2,))])  # forward_speed, dist to goal

        # TODO(ekl) this isn't really a proper gym spec
        self._spec = lambda: None
        self._spec.id = "Carla-v0"

        self.num_vehicle = NUM_VEHICLE

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

    def get_scenarios(self, choice):
        if choice == "1":
            self.config["server_map"] = "/Game/Carla/Maps/Town01"
            return DEFAULT_SCENARIO_TOWN1
        elif choice == "2":
            self.config["server_map"] = "/Game/Carla/Maps/Town02"
            return DEFAULT_SCENARIO_TOWN2

    def init_server(self):
        print("Initializing new Carla server...")
        # Create a new server process and start the client.
        self.server_port = 2000 #random.randint(10000, 60000)
        gpus = GPUtil.getGPUs()
        print('Get gpu:')
        if not self.config["render"] and ( gpus is not None and len(gpus) ) > 0:
            min_index = random.randint(0,len(gpus)-1)
            for i, gpu in enumerate(gpus):
                if gpu.load < gpus[min_index].load:
                    min_index = i

            self.server_process = subprocess.Popen(
                ("DISPLAY=:8 vglrun -d :7.{} {} "+self.config["server_map"]+" -windowed -ResX=800 -ResY=600 -carla-server -carla-world-port={}").format(min_index, SERVER_BINARY, self.server_port),
                shell=True, preexec_fn=os.setsid, stdout=open(os.devnull, "w"))
        else:
            self.server_process = subprocess.Popen(
                [SERVER_BINARY, self.config["server_map"],
                 "-windowed", "-ResX=800", "-ResY=600",
                 "-carla-server",
                 "-carla-world-port={}".format(self.server_port)],
                preexec_fn=os.setsid, stdout=open(os.devnull, "w"))
        live_carla_processes.add(os.getpgid(self.server_process.pid))

        # wait for carlar server to start
        time.sleep(15)

        self.actor_list = []
        self.client = carla.Client("localhost", self.server_port)
        
        
        
        
        #  Original in 0.8.2
        #for i in range(RETRIES_ON_ERROR):
        #    try:
        #        self.client = carla.Client("localhost", self.server_port)
                #self.client = CarlaClient("localhost", self.server_port)
        #        return self.client.ping()
        #    except Exception as e:
        #        print("Error connecting: {}, attempt {}".format(e, i))
        #        time.sleep(2)
            

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
        self.num_steps = [0]
        self.total_reward = [0] 
        self.prev_measurement = None
        self.prev_image = None
        self.episode_id = datetime.today().strftime("%Y-%m-%d_%H-%M-%S_%f")
        self.measurements_file = None
        
        # Create a CarlaSettings object. This object is a wrapper around
        # the CarlaSettings.ini file. Here we set the configuration we
        # want for the new episode.
        #  The setting do not work in Carla 0.9.0.
        settings = CarlaSettings()

        # If config["scenarios"] is a single scenario, then use it if it's an array of scenarios, randomly choose one and init
        #  no update_scenarios_parameter, it is from the old planner API.
        self.config = update_scenarios_parameter(self.config)

        #  the following block also does not work in 0.9.0.
        if isinstance(self.config["scenarios"],dict):
            self.scenario = self.config["scenarios"]
        else: #ininstance array of dict
            self.scenario = random.choice(self.config["scenarios"])
        assert self.scenario["city"] == self.city, (self.scenario, self.city)
        self.weather = random.choice(self.scenario["weather_distribution"])
        settings.set(
            SynchronousMode=True,
            SendNonPlayerAgentsInfo=True,
            NumberOfVehicles=self.scenario["num_vehicles"],
            NumberOfPedestrians=self.scenario["num_pedestrians"],
            WeatherId=self.weather)
        settings.randomize_seeds()
        print("-----> this is", settings)
        #  Create new camera in Carla_0.9.0.
        #client = carla.Client("localhost", self.server_port)
        #client.set_timeout(2000)
        world = self.client.get_world()
        cam_blueprint = world.get_blueprint_library().find('sensor.camera')
        camera = world.spawn_actor(cam_blueprint, GLOBAL_CAM_POS)
        print('camera at %s' % camera.get_location())  
        self.camera = camera
        camera.listen(lambda image: self.get_image(image))
        #wait the camera's launching time to get first image
        print("camera finished")
        time.sleep(3)
        
        #  Asynchronously camera test:
        #print('image000:', self.image)
        #time.sleep(5)
        #print('image111:', self.image)
        #time.sleep(5)
        #print('image222:', self.image)
        #settings.add_sensor(camera)
        
        #time.sleep(1000)
        #  Create new camera instead of the old API in the following block.
        #if self.config["use_depth_camera"]:
        #    camera1 = Camera("CameraDepth", PostProcessing="Depth")
        #    camera1.set_image_size(
        #        self.config["render_x_res"], self.config["render_y_res"])
        #    camera1.set_position(30, 0, 130)
        #    settings.add_sensor(camera1)	
        
        #camera2 = Camera("CameraRGB")
        #camera2.set_image_size(
        #    seslf.config["render_x_res"], self.config["render_y_res"])
        #camera2.set_position(30, 0, 130)
        #settings.add_sensor(camera2)

        # Setup start and end positions
        #  currently use exact number instead of the API in old planner.
        #scene = self.client.load_settings(settings)
        
        #  in python2, key is unicode, in python3, key is string
        #for x in POS_COOR_MAP:
        #    print(type(x))
        #    print(type(POS_COOR_MAP[x]))


        start_id = self.scenario["start_pos_id"]
        end_id = self.scenario["end_pos_id"]
        start_id = str(start_id)
        end_id = str(end_id)
        #start_id = str(start_id).decode("utf-8") # unicode is needed. this trans is for py2
        #end_id = str(end_id).decode("utf-8")
        
        
        POS_S = [[0] * 3] * self.num_vehicle
        POS_E = [[0] * 3] * self.num_vehicle
        POS_S[0] = POS_COOR_MAP[start_id]
        POS_E[0] = POS_COOR_MAP[end_id]
        
        world = self.client.get_world()
        testlib = world.get_blueprint_library()
        
        for i in range(self.num_vehicle):
            blueprints = world.get_blueprint_library().filter('vehicle')
            blueprint = random.choice(blueprints)
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
            transform = carla.Transform(
	        carla.Location(x=POS_S[i][0], y=POS_S[i][1], z=POS_S[i][2]),
	        carla.Rotation(yaw=0.0))
            print('spawning vehicle %r with %d wheels' % (blueprint.id, blueprint.get_attribute('number_of_wheels')))
            vehicle = world.try_spawn_actor(blueprint, transform)
            
            print('vehicle at %s' % vehicle.get_location())
            #print('vehicle at %s' % vehicle.get_velocity())
            
            self.actor_list.append(vehicle)
            #while True:
            #    s_time = time.time()
            #    print('vehicle at %s' % vehicle.get_location())       
            #    print('time: ', time.time()-s_time)
        print('All vehicles are created.')

        #scene = self.client.load_settings(settings)
        #print ("scene: ", scene)
        #time.sleep(1000)
        #positions = scene.player_start_spots
        #self.start_pos = positions[self.scenario["start_pos_id"]]
        #self.end_pos = positions[self.scenario["end_pos_id"]]
        #self.start_coord = [
        #    self.start_pos.location.x // 100, self.start_pos.location.y // 100]
        #self.end_coord = [
        #    self.end_pos.location.x // 100, self.end_pos.location.y // 100]
        #print(
        #    "Start pos {} ({}), end {} ({})".format(
        #        self.scenario["start_pos_id"], self.start_coord,
        #        self.scenario["end_pos_id"], self.end_coord))
        

        

        
        
        
        #  Need to print for multiple client
        self.start_pos = POS_S
        self.end_pos = POS_E
        self.start_coord = []
        self.end_coord = []
        self.py_measurement = {}
        self.prev_measurement = {}
        self.obs = []
        
        for i in range(self.num_vehicle):
            self.start_coord.append([
                self.start_pos[i][0] // 100, self.start_pos[i][1] // 100])
            self.end_coord.append([
                self.end_pos[i][0] // 100, self.end_pos[i][1] // 100])
            
            print(
                "Client {} start pos {} ({}), end {} ({})".format(
                    i, self.start_pos[i], self.start_coord[i],
                    self.end_pos[i], self.end_coord[i]))
             
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
            
            obs = self.encode_obs(self.image, self.py_measurement[vehcile_name], i)
            self.obs_dict[vehcile_name] = obs
            
        return self.obs_dict

    def get_image(self, image):
        #print(image)
        #print('GET IMAGE >>>>>')
        self.original_image = image
        self._parse_image(image) # py_game render use
        self.image = self.preprocess_image(image)
        #print('FINISH IMAGE <<<<<')
        
    def encode_obs(self, image, py_measurements, vehcile_number):
        
        assert self.config["framestack"] in [1, 2]
        # currently, the image is generated asynchronously
        prev_image = self.prev_image
        self.prev_image = image
        if prev_image is None:
            prev_image = image
        if self.config["framestack"] == 2:
            #image = np.concatenate([prev_image, image], axis=2)
            image = np.concatenate([prev_image, image])
        obs = (
            'Vehcile number: ',
            vehcile_number,
            image,
            COMMAND_ORDINAL[py_measurements["next_command"]],
            [py_measurements["forward_speed"],
             py_measurements["distance_to_goal"]])
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
                obs, reward, done, info = self._step(action_dict[action], actor_num)
                
                vehcile_name = 'Vehcile'
                vehcile_name +=str(actor_num)
                actor_num += 1
                obs_dict[vehcile_name] = obs
                reward_dict[vehcile_name] = reward
                done_dict[vehcile_name] = done
                info_dict[vehcile_name] = info
                    
            return obs_dict, reward_dict, done_dict, info_dict
        except Exception:
            print(
                "Error during step, terminating episode early",
                traceback.format_exc())
            self.clear_server_state()
            return (self.last_obs, 0.0, True, {})

    def _step(self, action, i):
        if self.config["discrete_actions"]:
            action = DISCRETE_ACTIONS[int(action)]
        assert len(action) == 2, "Invalid action {}".format(action)
        if self.config["squash_action_logits"]:
            forward = 2 * float(sigmoid(action[0]) - 0.5)
            throttle = float(np.clip(forward, 0, 1))
            brake = float(np.abs(np.clip(forward, -1, 0)))
            steer = 2 * float(sigmoid(action[1]) - 0.5)
        else:
            throttle = float(np.clip(action[0], 0, 1))
            brake = float(np.abs(np.clip(action[0], -1, 0)))
            steer = float(np.clip(action[1], -1, 1))
        reverse = False
        hand_brake = False 
        if self.config["verbose"]:
            print(
                "steer", steer, "throttle", throttle, "brake", brake,
                "reverse", reverse)

        #  send control
        if self.config['manual_control']: 
            if i == 0:
                #pygame need this
                self._display = pygame.display.set_mode(
                        (800, 600),
                        pygame.HWSURFACE | pygame.DOUBLEBUF)
                logging.debug('pygame started')
         
                control1 = self._get_keyboard_control1(pygame.key.get_pressed())
                self.actor_list[i].apply_control(control1)
                self._on_render()
            else:
                self._display = pygame.display.set_mode(
                        (800, 600),
                        pygame.HWSURFACE | pygame.DOUBLEBUF)
                logging.debug('pygame started')
         
                control2 = self._get_keyboard_control2(pygame.key.get_pressed())
                self.actor_list[i].apply_control(control2)
                self._on_render()
        elif self.config["auto_control"]:    
            self.actor_list[i].set_autopilot()
        else:
            self.actor_list[i].apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake, hand_brake=hand_brake, reverse=reverse))

        
        # Process observations
        py_measurements = self._read_observation(i)
        
        if self.config["verbose"]:
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
        reward = compute_reward(
            self, self.prev_measurement[vehcile_name], py_measurements)
        
        #  update num_steps and total_reward lists if next car comes
        if i == len(self.num_steps):
            self.num_steps.append(0)
        if i == len(self.total_reward):
            self.total_reward.append(0)
        
        self.total_reward[i] += reward
        py_measurements["reward"] = reward
        py_measurements["total_reward"] = self.total_reward
        done = (self.num_steps[i] > MAX_STEP or #self.scenario["max_steps"] or
                py_measurements["next_command"] == "REACH_GOAL")# or
                #(self.config["early_terminate_on_collision"] and
                # collided_done(py_measurements)))
        py_measurements["done"] = done
        
        self.prev_measurement[vehcile_name] = py_measurements
        self.num_steps[i] += 1

        # Write out measurements to file
        if i == self.num_vehicle - 1:#print all cars measurement
            if CARLA_OUT_PATH:
                if not self.measurements_file:
                    self.measurements_file = open(
                        os.path.join(
                            CARLA_OUT_PATH,
                            "measurements_{}.json".format(self.episode_id)),
                        "w")
                self.measurements_file.write(json.dumps(self.py_measurement))
                self.measurements_file.write("\n")
                #if done:
                #    self.measurements_file.close()
                #    self.measurements_file = None
                #    if self.config["convert_images_to_video"]:
                #        self.images_to_video()
        
        return (
            self.encode_obs(self.image, py_measurements, i), reward, done,
            py_measurements)

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
        #if keys[K_q]:
        #    self._is_on_reverse = not self._is_on_reverse
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
        #if keys[K_q]:
        #    self._is_on_reverse = not self._is_on_reverse
        #if keys[K_p]:
        #    self._autopilot_enabled = not self._autopilot_enabled
        #control.reverse = self._is_on_reverse
        return control

    def images_to_video(self):
        videos_dir = os.path.join(CARLA_OUT_PATH, "Videos")
        if not os.path.exists(videos_dir):
            os.makedirs(videos_dir)
        ffmpeg_cmd = (
            "ffmpeg -loglevel -8 -r 60 -f image2 -s {x_res}x{y_res} "
            "-start_number 0 -i "
            "{img}_%04d.jpg -vcodec libx264 {vid}.mp4 && rm -f {img}_*.jpg "
        ).format(
            x_res=self.config["render_x_res"],
            y_res=self.config["render_y_res"],
            vid=os.path.join(videos_dir, self.episode_id),
            img=os.path.join(CARLA_OUT_PATH, "CameraRGB", self.episode_id))
        print("Executing ffmpeg command", ffmpeg_cmd)
        subprocess.call(ffmpeg_cmd, shell=True)

    def preprocess_image(self, image):
        if self.config["use_depth_camera"]:
            assert self.config["use_depth_camera"]
            data = (image.raw_data - 0.5) * 2
            data = data.reshape(
                self.config["render_y_res"], self.config["render_x_res"], 1)
            data = cv2.resize(
                data, (self.config["x_res"], self.config["y_res"]),
                interpolation=cv2.INTER_AREA)
            data = np.expand_dims(data, 2)
        else:
            #data = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            #data = np.reshape(data, (self.config["render_y_res"], self.config["render_x_res"], 3))
            data = np.reshape(image.raw_data,
                (self.config["render_y_res"], self.config["render_x_res"],4))
            data = np.resize(
                data, (self.config["x_res"], self.config["y_res"]))
            data = (data.astype(np.float32) - 128) / 128
        return data

    def _save_to_disk(self, image):

        filename = '_images/{:0>6d}_{:s}.png'.format(image.frame_number, image.type)
    
        try:
            from PIL import Image as PImage
        except ImportError:
            raise RuntimeError(
                'cannot import PIL, make sure pillow package is installed')
        
        image = PImage.frombytes(
            mode='RGBA',
            size=(image.width, image.height),
            data=image.raw_data,
            decoder_name='raw')
        color = image.split()
        image = PImage.merge("RGB", color[2::-1])


        out_dir = os.path.join(CARLA_OUT_PATH, filename)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_file = os.path.join(
            out_dir,
            "{}_{:>04}.jpg".format(self.episode_id, self.num_steps))
        scipy.misc.imsave(out_file, image.raw_data)
        

        
       
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

        
        #cur = measurements.player_measurements
        #if self.config["enable_planner"]:
        #    next_command = COMMANDS_ENUM[
        #        self.planner.get_next_command(
        #            [cur.transform.location.x, cur.transform.location.y,
        #             GROUND_Z],
        #            [cur.transform.orientation.x, cur.transform.orientation.y,
        #             GROUND_Z],
        #            [self.end_pos.location.x, self.end_pos.location.y,
        #             GROUND_Z],
        #            [self.end_pos.orientation.x, self.end_pos.orientation.y,
        #             GROUND_Z])
        #    ]
        #else:
        #    next_command = "LANE_FOLLOW"

        #print('start calculate distance')
        #s_dis = time.time()
        #  A simple planner
        current_x = self.actor_list[i].get_location().x
        current_y = self.actor_list[i].get_location().y

        print('start calculate distance')
        s_dis = time.time()
        distance_to_goal_euclidean = float(np.linalg.norm(
            [current_x - self.end_pos[i][0],
             current_y - self.end_pos[i][1]]) / 100)
        
        distance_to_goal = distance_to_goal_euclidean
        diff_x =  abs(current_x - self.end_pos[i][0])
        diff_y =  abs(current_y - self.end_pos[i][1])
        if diff_x < 1 and diff_y < 1:
            next_command = "REACH_GOAL"
        else:
            next_command = "LANE_FOLLOW"
         
        
        #print('calculate distance finished')
        #print('cal dist time: ', time.time() - s_dis)

        #if next_command == "REACH_GOAL":
        #    distance_to_goal = 0.0  # avoids crash in planner
        #else:
        #    distance_to_goal = distance_to_goal_euclidean
        #elif self.config["enable_planner"]:
        #    distance_to_goal = self.planner.get_shortest_path_distance(
        #        [cur.transform.location.x, cur.transform.location.y, GROUND_Z],
        #        [cur.transform.orientation.x, cur.transform.orientation.y,
        #         GROUND_Z],
        #        [self.end_pos.location.x, self.end_pos.location.y, GROUND_Z],
        #        [self.end_pos.orientation.x, self.end_pos.orientation.y,
        #         GROUND_Z]) / 100
        #else:
        #    distance_to_goal = -1

        #print('store py: ')
        #s_dis = time.time()
        py_measurements = {
            "episode_id": self.episode_id,
            "step": self.num_steps,
            "x": current_x,
            "y": current_y,
            #"x_orient": cur.transform.orientation.x,
            #"y_orient": cur.transform.orientation.y,
            "forward_speed": 0,
            "distance_to_goal": distance_to_goal,#use planner
            "distance_to_goal_euclidean": distance_to_goal_euclidean,
            #"collision_vehicles": cur.collision_vehicles,
            #"collision_pedestrians": cur.collision_pedestrians,
            #"collision_other": cur.collision_other,
            #"intersection_offroad": cur.intersection_offroad,
            #"intersection_otherlane": cur.intersection_otherlane,
            #"weather": self.weather,
            #"map": self.config["server_map"],
            "start_coord": self.start_coord,
            "end_coord": self.end_coord,
            #"current_scenario": self.scenario,
            #"x_res": self.config["x_res"],
            #"y_res": self.config["y_res"],
            #"num_vehicles": self.scenario["num_vehicles"],
            #"num_pedestrians": self.scenario["num_pedestrians"],
            "max_steps": 1000, # set 1000 now. self.scenario["max_steps"],
            "next_command": next_command,
        }
        
        save_to_disk(self.original_image)
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
        print('calculate distance finished')
        print('cal dist time: ', time.time() - s_dis)
        return py_measurements


def compute_reward_corl2017(env, prev, current):
    reward = 0.0

    cur_dist = current["distance_to_goal"]

    prev_dist = prev["distance_to_goal"]

    if env.config["verbose"]:
        print("Cur dist {}, prev dist {}".format(cur_dist, prev_dist))

    # Distance travelled toward the goal in m
    reward += np.clip(prev_dist - cur_dist, -10.0, 10.0)

    # Change in speed (km/h)
    reward += 0.05 * (current["forward_speed"] - prev["forward_speed"])
    

    #  no collision and sidewarlk now.
    # New collision damage
    #reward -= .00002 * (
    #    current["collision_vehicles"] + current["collision_pedestrians"] +
    #    current["collision_other"] - prev["collision_vehicles"] -
    #    prev["collision_pedestrians"] - prev["collision_other"])

    # New sidewalk intersection
    #reward -= 2 * (
    #    current["intersection_offroad"] - prev["intersection_offroad"])

    # New opposite lane intersection
    #reward -= 2 * (
    #    current["intersection_otherlane"] - prev["intersection_otherlane"])

    return reward


def compute_reward_custom(env, prev, current):
    reward = 0.0

    cur_dist = current["distance_to_goal"]
    prev_dist = prev["distance_to_goal"]

    if env.config["verbose"]:
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


def compute_reward(env, prev, current):
    return REWARD_FUNCTIONS[env.config["reward_function"]](
        env, prev, current)


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
    collided = (
        m["collision_vehicles"] > 0 or m["collision_pedestrians"] > 0 or
        m["collision_other"] > 0)
    return bool(collided or m["total_reward"] < -100)


if __name__ == "__main__":
    #  Episode for loop  
    #from multi_env import MultiCarlaEnv
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '--scenario',
        default = '1',
        help='print debug information')

    argparser.add_argument(
        '--config',
        default = 'config.json',
        help='print debug information')

    argparser.add_argument(
        '--map',
        default = 'Town01',
        help='print debug information')


    args = argparser.parse_args()

    POS_COOR_MAP = json.load(open("POS_COOR/pos_cordi_map_town1.txt"))
    
    for _ in range(1):
        #  Initialize server and clients.
        env = MultiCarlaEnv(args)
        print('env finished')
        obs = env.reset() 
        print('obs infor:')
        print(obs)
        #time.sleep(1000) #  test use

        start = time.time()
        done = False
        i = 0

        #  Initialize total reward dict.
        total_reward_dict = {}
        for n in range(NUM_VEHICLE):
            vehcile_name = 'Vehcile'
            vehcile_name += str(n)
            total_reward_dict[vehcile_name] = 0 

        
        #  3 in action_list means go straight. 
        action_list = {
            'Vehcile0' : 3,
            #'Vehcile1' : 3,
        }

        all_done = False
        while not all_done:
            i += 1
            if ENV_CONFIG["discrete_actions"]:
                obs, reward, done, info = env.step(action_list)
            else:
                obs, reward, done, info = env.step([0, 1, 0])
            
            for t in total_reward_dict:
                total_reward_dict[t] += reward[t]
            
            print("Step", i, "rew", reward, "total", total_reward_dict, "done", done)

            #  Test whether all vehicles have finished.
            done_temp = True
            for d in done:
                done_temp  = done_temp and done[d]
            all_done = done_temp
        
        print("{} fps".format(100 / (time.time() - start)))
    
