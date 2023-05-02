"""Global constant values."""

import math

import carla
from carla_gym.carla_api.PythonAPI.agents.navigation.local_planner import RoadOption

# of env
DEFAULT_MULTIENV_CONFIG = {
    "scenarios": {
        "scenario1": {
            "max_steps": 500,
            "town": "Town01",
            "objects": {
                "vehicle1": {
                    "start_x": 115,
                    "start_y": 132,
                    "start_z": 0.5,
                    "yaw": 0,
                    "end_x": 125,
                    "end_y": 142,
                    "end_z": 0.5,
                    "model": "vehicle.lincoln.mkz_2017",
                },
                "vehicle2": {
                    "start_x": 107,
                    "start_y": 133.5,
                    "start_z": 0.5,
                    "yaw": 0,
                    "end_x": 117,
                    "end_y": 143.5,
                    "end_z": 0.5,
                    "model": "vehicle.lincoln.mkz_2017",
                },
            },
        },
        "scenario2": {
            "max_steps": 500,
            "town": "Town01",
            "objects": {
                "vehicle1": {
                    "start_x": 115,
                    "start_y": 132,
                    "start_z": 0.5,
                    "yaw": 0,
                    "end_x": 125,
                    "end_y": 142,
                    "end_z": 0.5,
                    "model": "vehicle.lincoln.mkz_2017",
                },
                "vehicle2": {
                    "start_x": 107,
                    "start_y": 133.5,
                    "start_z": 0.5,
                    "yaw": 0,
                    "end_x": 117,
                    "end_y": 143.5,
                    "end_z": 0.5,
                    "model": "vehicle.lincoln.mkz_2017",
                },
            },
        },
    },
    "actors": {
        "vehicle1": {
            "enable_planner": True,
            "render": True,
            "framestack": 1,  # note: only [1, 2] currently supported
            "early_terminate_on_collision": True,
            "squash_action_logits": False,
            "manual_control": False,
            "auto_control": False,
            "camera_type": "rgb",
            "camera_position": 0,
            "collision_sensor": True,
            "lane_sensor": True,
            "send_measurements": False,
            "log_images": False,
            "log_measurements": False,
        }
    },
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

WEATHERS = {
    "Default": carla.WeatherParameters.Default,
    "ClearNoon": carla.WeatherParameters.ClearNoon,
    "CloudyNoon": carla.WeatherParameters.CloudyNoon,
    "WetNoon": carla.WeatherParameters.WetNoon,
    "WetCloudyNoon": carla.WeatherParameters.WetCloudyNoon,
    "MidRainyNoon": carla.WeatherParameters.MidRainyNoon,
    "HardRainNoon": carla.WeatherParameters.HardRainNoon,
    "SoftRainNoon": carla.WeatherParameters.SoftRainNoon,
    "ClearSunset": carla.WeatherParameters.ClearSunset,
    "CloudySunset": carla.WeatherParameters.CloudySunset,
    "WetSunset": carla.WeatherParameters.WetSunset,
    "WetCloudySunset": carla.WeatherParameters.WetCloudySunset,
    "MidRainSunset": carla.WeatherParameters.MidRainSunset,
    "HardRainSunset": carla.WeatherParameters.HardRainSunset,
    "SoftRainSunset": carla.WeatherParameters.SoftRainSunset,
}

TEST_WEATHERS = [
    "ClearNoon",
    "WetNoon",
    "HardRainNoon",
    "ClearSunset",
    "WetSunset",
    "WetCloudySunset",
    "MidRainSunset",
    "HardRainSunset",
    "SoftRainSunset",
]
TRAIN_WEATHERS = [
    "CloudyNoon",
    "WetCloudyNoon",
    "MidRainyNoon",
    "SoftRainNoon",
    "CloudySunset",
]

PAPER_TEST_WEATHERS = ["ClearNoon", "WetCloudyNoon", "HardRainNoon", "ClearSunset"]
PAPER_TRAIN_WEATHERS = ["CloudyNoon", "SoftRainSunset"]
