"""
follow_leading_vehicle.py: Example of creating a custom environment, also a demonstration of how to use the manual_control.
__author__: Morphlng
"""

import random
from macad_gym.envs import MultiCarlaEnv

"""
    This is a scenario extracted from Carla/scenario_runner (https://github.com/carla-simulator/scenario_runner/blob/master/srunner/scenarios/follow_leading_vehicle.py).

    The configuration below contains everything you need to customize your own scenario in Macad-Gym.
"""
configs = {
    "scenarios": {
        "map": "Town01",
        "actors": {
            "car1": {
                "start": [107, 133, 0.5],
                "end": [300, 133, 0.5],
            },
            "car2": {
                "start": [115, 133, 0.5],
                "end": [310, 133, 0.5],
            }
        },
        "num_vehicles": 0,
        "num_pedestrians": 0,
        "weather_distribution": [0],
        "max_steps": 500
    },
    "env": {
        "server_map": "/Game/Carla/Maps/Town01",
        "render": False,
        "render_x_res": 800,    # For both Carla-Server and Manual-Control
        "render_y_res": 600,
        "x_res": 84,            # Used for camera sensor view size
        "y_res": 84,
        "framestack": 1,
        "discrete_actions": True,
        "squash_action_logits": False,
        "verbose": False,
        "use_depth_camera": False,
        "send_measurements": False,
        "enable_planner": True,
        "spectator_loc": [70, -125, 9],
        "sync_server": True,
        "fixed_delta_seconds": 0.05,
    },
    "actors": {
        "car1": {
            "type": "vehicle_4W",
            "enable_planner": True,
            "convert_images_to_video": False,
            "early_terminate_on_collision": True,
            "reward_function": "corl2017",
            "scenarios": "FOLLOWLEADING_TOWN1_CAR1",
            
            # When "auto_control" is True,
            # starts the actor using auto-pilot.
            # Allows manual control take-over on
            # pressing Key `p` on the PyGame window
            # if manual_control is also True
            "manual_control": True,
            "auto_control": True,

            "camera_type": "rgb",
            "collision_sensor": "on",
            "lane_sensor": "on",
            "log_images": False,
            "log_measurements": False,
            "render": True,
            "x_res": 84,    # Deprecated, kept for backward compatibility
            "y_res": 84,
            "use_depth_camera": False,
            "send_measurements": False,
        },
        "car2": {
            "type": "vehicle_4W",
            "enable_planner": True,
            "convert_images_to_video": False,
            "early_terminate_on_collision": True,
            "reward_function": "corl2017",
            "scenarios": "FOLLOWLEADING_TOWN1_CAR2",
            "manual_control": False,
            "auto_control": True,
            "camera_type": "rgb",
            "collision_sensor": "on",
            "lane_sensor": "on",
            "log_images": False,
            "log_measurements": False,
            "render": True,
            "x_res": 84,
            "y_res": 84,
            "use_depth_camera": False,
            "send_measurements": False,
        }
    },
}


class FollowLeadingVehicle(MultiCarlaEnv):
    """A two car Multi-Agent Carla-Gym environment"""

    def __init__(self):
        self.configs = configs
        super(FollowLeadingVehicle, self).__init__(self.configs)


if __name__ == "__main__":
    env = FollowLeadingVehicle()

    for ep in range(2):
        obs = env.reset()

        total_reward_dict = {}
        action_dict = {}

        env_config = configs["env"]
        actor_configs = configs["actors"]
        for actor_id in actor_configs.keys():
            total_reward_dict[actor_id] = 0
            if env_config['discrete_actions']:
                # take random action
                # The action will be ignored if the actor is controlled by auto_control/manual_control
                action_dict[actor_id] = random.randint(0, 8)
            else:
                action_dict[actor_id] = [1, 0]  # test values

        i = 0
        done = {"__all__": False}
        while not done["__all__"]:
            i += 1
            obs, reward, done, info = env.step(action_dict)

            for actor_id in total_reward_dict.keys():
                total_reward_dict[actor_id] += reward[actor_id]

            print("Episode: {}, Step: {}, Reward: {}, Done: {}".format(
                ep, i, total_reward_dict, done))

    env.close()
