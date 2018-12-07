"""
multi_env.py: Multi-actor environment interface for CARLA-Gym
Should support two modes of operation. See CARLA-Gym developer guide for more info
__author__: PP, BP
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import atexit
from datetime import datetime
import glob

import json
import os
import random
import signal
import subprocess
import sys
import time
import traceback

import GPUtil
from gym.spaces import Box, Discrete, Tuple
import pygame
try:
    import scipy.misc
except Exception:
    pass

sys.path.append(
    glob.glob(f'**/**/PythonAPI/lib/carla-*{sys.version_info.major}.'
              f'{sys.version_info.minor}-linux-x86_64.egg')[0])
import carla
from env.multi_actor_env import *
from env.core.sensors.utils import preprocess_image
from env.core.sensors.camera_list import CameraList
from env.core.vehicle_manager import VehicleManager

from env.carla.scenarios import *
from env.carla.reward import *
from env.carla.carla.planner import *

# Set this where you want to save image outputs (or empty string to disable)
CARLA_OUT_PATH = os.environ.get("CARLA_OUT", os.path.expanduser("~/carla_out"))
if CARLA_OUT_PATH and not os.path.exists(CARLA_OUT_PATH):
    os.makedirs(CARLA_OUT_PATH)

# Set this to the path of your Carla binary
SERVER_BINARY = os.environ.get(
    "CARLA_SERVER", os.path.expanduser("~/software/CARLA_0.9.1/CarlaUE4.sh"))

assert os.path.exists(SERVER_BINARY)

# Number of max step
MAX_STEP = 1000

DEFAULT_MULTIENV_CONFIG = {
    "0": {
        "log_images": True,
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
        "collision_sensor": "on",  #off
        "lane_sensor": "on",  #off
        "server_process": False,
        "send_measurements": False
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


class MultiCarlaEnv(MultiActorEnv):
    """Carla environment, similar to the World class of manual_control.py from Carla.

    Args:
        A list of config files for actors.

    """

    def __init__(self, actor_configs=DEFAULT_MULTIENV_CONFIG):

        self.actor_configs = actor_configs

        # Set attributes as in gym's specs
        self.reward_range = (-np.inf, np.inf)
        self.metadata = {'render.modes': 'human'}

        # Get general/same config for actors.
        general_config = self.actor_configs[str(0)]
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
        self.use_depth_camera = False  #!!test
        self.camera_list = CameraList(CARLA_OUT_PATH)
        # self.config["server_map"] = "/Game/Carla/Maps/" + args.map  # For arg map.

        # Needed by agents
        if self.discrete_actions:
            self.action_space = Discrete(len(DISCRETE_ACTIONS))
        else:
            self.action_space = Box(-1.0, 1.0, shape=(2, ))

        if self.use_depth_camera:
            image_space = Box(
                -1.0, 1.0, shape=(self.y_res, self.x_res, 1 * self.framestack))
        elif self.actor_configs["0"]["send_measurements"]:
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
            self.observation_space = Box(0.0, 255.0,
                                         shape=(self.y_res, self.x_res, 3 * self.framestack))

        # Set planner list for actors.
        self.planner_list = []
        for k in self.actor_configs:
            config = self.actor_configs[k]
            # config["scenarios"] = self.get_scenarios(args.scenario, config)
            if config["enable_planner"]:
                self.planner_list.append(Planner(self.city))

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

        self.last_reward = []

        self.vehicle_manager_list = []
        self.scenario_list = []

    def get_scenarios(self, choice):
        if choice == "DEFAULT_SCENARIO_TOWN1":
            return DEFAULT_SCENARIO_TOWN1
        elif choice == "DEFAULT_SCENARIO_TOWN1_2":
            return DEFAULT_SCENARIO_TOWN1_2
        elif choice == "DEFAULT_SCENARIO_TOWN2":
            return DEFAULT_SCENARIO_TOWN2
        elif choice == "TOWN1_STRAIGHT":
            return TOWN1_STRAIGHT
        elif choice == "CURVE_TOWN1":
            return CURVE_TOWN1
        elif choice == "CURVE_TOWN2":
            return CURVE_TOWN2
        elif choice == "DEFAULT_CURVE_TOWN1":
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
                    # "-carla-settings=/home/fastisu/Documents/CARLA-Gym/bo_CS.ini",
                    "-benchmark -fps=10"
                    "-carla-server",
                    "-carla-world-port={}".format(self.server_port)
                ],
                preexec_fn=os.setsid,
                stdout=open(os.devnull, "w"))
        live_carla_processes.add(os.getpgid(self.server_process.pid))

        #  wait for carla server to start
        time.sleep(10)

        # Start client
        self.client = carla.Client("localhost", self.server_port)

    def clean_world(self):
        """Destroy all sensors after the program terminates.

        Returns:
            N/A

        """
        for v_m in env.vehicle_manager_list:
            if v_m._config["camera"] == "on":
                v_m._camera_manager.sensor.destroy()
            if v_m._config["collision_sensor"] == "on":
                v_m.__collision_sensor.sensor.destroy()
            if v_m._config["lane_sensor"] == "on":
                v_m._lane_invasion_sensor.sensor.destroy()
            v_m._vehicle.destroy()


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

    def __del__(self):
        self.clear_server_state()

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

    def _reset(self):
        """Initialize actors.

        Get poses for vehicle actors and spawn them.
        Spawn all sensors as needed.

        Returns:
            dict: observation dictionaries for actors.
        """

        # If config contains a single scenario, then use it,
        # if it's an array of scenarios, randomly choose one and init
        for k in self.actor_configs:
            config = self.actor_configs[k]
            scenario = self.get_scenarios(config["scenarios"])
            if isinstance(scenario, dict):
                self.scenario_list.append(scenario)
            else:  # instance array of dict
                self.scenario_list.append(random.choice(scenario))

        NUM_VEHICLE = len(self.scenario_list)
        self.num_vehicle = NUM_VEHICLE
        self.num_steps = [0] * self.num_vehicle
        self.total_reward = [0] * self.num_vehicle
        self.prev_measurement = None
        self.episode_id = datetime.today().strftime("%Y-%m-%d_%H-%M-%S_%f")
        self.measurements_file = None

        self.start_coord = []
        self.end_coord = []
        self.py_measurement = {}
        self.prev_measurement = {}
        self.obs = []

        self.last_reward = [0] * self.num_vehicle
        start_transform = []
        end_transform = []
        for i in range(self.num_vehicle):
            scenario = self.scenario_list[i]
            # str(start_id).decode("utf-8") # for py2
            s_id = str(scenario["start_pos_id"])
            e_id = str(scenario["end_pos_id"])
            pos_s = self.pos_coor_map[s_id]
            pos_e = self.pos_coor_map[e_id]

            s_loc = carla.Location(x=pos_s[0], y=pos_s[1], z=pos_s[2])
            e_loc = carla.Location(x=pos_e[0], y=pos_e[1], z=pos_e[2])
            s_rot = carla.Rotation(pitch=0, yaw=0, roll=0)
            e_rot = carla.Rotation(pitch=0, yaw=0, roll=0)

            start_transform.append(carla.Transform(s_loc, s_rot))
            end_transform.append(carla.Transform(e_loc, e_rot))

            self.start_coord.append(
                [pos_s[0] // 100, pos_s[1] // 100])
            self.end_coord.append(
                [pos_e[0] // 100, pos_e[1] // 100])
            print("Client {} start pos {} ({}), end {} ({})".format(
                i, pos_s[i], self.start_coord[i], pos_e[i],
                self.end_coord[i]))

        world = self.client.get_world()
        self.world = world
        self.weather = [
            world.get_weather().cloudyness,
            world.get_weather().precipitation,
            world.get_weather().precipitation_deposits,
            world.get_weather().wind_intensity
        ]

        for i in range(self.num_vehicle):
            config = self.actor_configs[str(i)]
            # Spawn vehicle
            vehicle_manager = VehicleManager()
            transform = start_transform[i]
            vehicle_manager.set_world(self.world)
            vehicle_manager.set_config(config)
            vehicle_manager.set_vehicle(transform)
            self.vehicle_manager_list.append(vehicle_manager) # maybe do not need list
            self.vehicle_manager_list[i].set_scenario(scenario)

        print('Environment initialized with requested actors.')

        i = 0
        for v_m in self.vehicle_manager_list:
            v_m.set_transform(end_transform[i])
            py_mt = v_m.read_observation()
            vehicle_name = 'Vehicle'
            vehicle_name += str(i)
            image = preprocess_image(v_m._camera_manager.image, config)
            obs = v_m.encode_obs(image, py_mt)
            self.obs_dict[vehicle_name] = obs

            self.py_measurement[vehicle_name] = py_mt
            self.prev_measurement[vehicle_name] = py_mt
            i = i + 1

        return self.obs_dict

    def step(self, action_dict):
        """One step for actors in simulation

        Args:
            action_dict (dict): actions represented by integers.

        Returns:
            dict: observation dict.
            dict: current reward dict.
            dict: bools indicate whether vehicles finish.
            dict: measurement dict.
        """
        try:
            obs_dict = {}
            reward_dict = {}
            done_dict = {}
            info_dict = {}

            actor_num = 0
            done_dict["__all__"] = False
            # TODO: The loop iter should go over self.agents not action_dict
            for action in action_dict:
                obs, reward, done, info = self._step(action_dict[action],
                                                     actor_num)
                vehicle_name = 'Vehicle'
                vehicle_name += str(actor_num)
                actor_num += 1
                obs_dict[vehicle_name] = obs
                reward_dict[vehicle_name] = reward
                done_dict[vehicle_name] = done
                if sum(list(done_dict.values())) == len(action_dict):
                    done_dict["__all__"] = True
                info_dict[vehicle_name] = info
            return obs_dict, reward_dict, done_dict, info_dict
        except Exception:
            print("Error during step, terminating episode early",
                  traceback.format_exc())
            self.clear_server_state()
            return self.last_obs, 0.0, True, {}

    def _step(self, action, i):
        """sub func of step()

        Send control to actor_list[i].
        Get its measurement.
        Compute its reward.

        Args:
            action (list): a len-two list, e.g., [1.0, 0.0].
            i: index from actor_list.

        Returns:
            dict: observation data.
            float: current reward for actor actor_list[i].
            bool: whether the actor gets to the destination.
            dict: measurement data.
        """
        ctrl_args = []
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

        ctrl_args.extend([throttle, steer, brake])
        ctrl_args.append(hand_brake)
        ctrl_args.append(reverse)

        v_m = self.vehicle_manager_list[i]
        v_m.apply_control(ctrl_args)

        # Process observations
        vehicle_measure = v_m.read_observation()
        py_measurements = {
            "episode_id": self.episode_id,
            "step": self.num_steps,
            "weather": self.weather,  # random.choice(self.scenario["weather_distribution"]),
            "start_coord": self.start_coord[i],
            "end_coord": self.end_coord[i],
            "x_res": self.x_res,
            "y_res": self.y_res,
            "num_vehicles": self.scenario_list[i]["num_vehicles"],
            "num_pedestrians": self.scenario_list[i]["num_pedestrians"],
            "max_steps": 1000,  #  set 1000 now. self.scenario["max_steps"],
        }
        py_measurements.update(vehicle_measure)

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
        config = self.actor_configs[str(i)]
        flag = config["reward_function"]
        cmpt_reward = Reward()
        vehicle_name = 'Vehicle'
        vehicle_name += str(i)
        reward = cmpt_reward.compute_reward(
            self.prev_measurement[vehicle_name], py_measurements, flag)

        self.last_reward[i] = reward  # to make the previous_rewards in py_measurements
        self.total_reward[i] += reward
        py_measurements["reward"] = reward
        py_measurements["total_reward"] = self.total_reward
        done = (
            self.num_steps[i] > MAX_STEP or  # self.scenario["max_steps"] or
            py_measurements["next_command"] == "REACH_GOAL")  # or
        # (self.config["early_terminate_on_collision"] and
        # collided_done(py_measurements)))
        py_measurements["done"] = done

        self.prev_measurement[vehicle_name] = py_measurements
        self.num_steps[i] += 1

        # Write out measurements to file
        if i == self.num_vehicle - 1:  # print all cars measurement
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
                    # if self.config["convert_images_to_video"] and (not self.video):
                    #    self.images_to_video()
                    #    self.video = Trueseg_city_space

        image = self.vehicle_manager_list[i]._camera_manager.image
        image = preprocess_image(image, config)

        return (v_m.encode_obs(image, py_measurements), reward, done,
                py_measurements)





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
    v = 0
    action_dict = {}
    for k in measurements:
        m = measurements[k]
        command = m["next_command"]
        name = 'Vehicle' + str(v)
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
        # Test for discrete actions:
        if not is_discrete_actions:
            action_dict[name] = [1, 0]
        v = v + 1
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
        #  Initialize server and clients.
        # env = MultiCarlaEnv(args)
        multi_env_config = json.load(open(args.config))

        env = MultiCarlaEnv(multi_env_config)
        obs = env.reset()
        print("obs, finished")

        total_vehicle = env.num_vehicle

        #  Initialize total reward dict.
        total_reward_dict = {}
        for n in range(total_vehicle):
            vehicle_name = 'Vehicle'
            vehicle_name += str(n)
            total_reward_dict[vehicle_name] = 0

        #  Initialize all vehicles' action to be 3
        action_dict = {}
        for v in range(total_vehicle):
            vehicle_name = 'Vehicle' + str(v)
            if env.discrete_actions:
                action_dict[vehicle_name] = 3
            else:
                action_dict[vehicle_name] = [1, 0]  # test number

        # server_clock = pygame.time.Clock()
        # print(server_clock.get_fps())

        start = time.time()
        i = 0
        # while not done["__all__"]:
        while i < 10:  # TEST
            i += 1
            obs, reward, done, info = env.step(action_dict)
            action_dict = get_next_actions(info, env.discrete_actions)
            print(reward)
            #time.sleep(100)
            for t in total_reward_dict:
                total_reward_dict[t] += reward[t]
            print("Step", i, "rew", reward, "total", total_reward_dict, "done",
                  done)

            # Set done[__all__] for env termination when all agents are done
            done_temp = True
            for d in done:
                done_temp = done_temp and done[d]
            done["__all__"] = done_temp
            time.sleep(0.1)

        print("{} fps".format(i / (time.time() - start)))

        # Clean actors in world
        # env.clean_world()

        # Set record options before set_sensor
        env.camera_list.save_images_to_disk()

