#!/usr/bin/env python
import time

from macad_gym.carla.multi_env import MultiCarlaEnv


class StopSign3CarTown03(MultiCarlaEnv):
    """A 4-way signalized intersection Multi-Agent Carla-Gym environment"""
    def __init__(self):
        self.configs = {
            "scenarios": "SSUI3C_TOWN3",
            "env": {
                "server_map": "/Game/Carla/Maps/Town03",
                "render": True,
                "render_x_res": 800,
                "render_y_res": 600,
                "x_res": 168,
                "y_res": 168,
                "framestack": 1,
                "discrete_actions": True,
                "squash_action_logits": False,
                "verbose": False,
                "use_depth_camera": False,
                "send_measurements": False,
                "enable_planner": True,
                "spectator_loc": [140, 68, 9],
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
                    "scenarios": "SSUI3C_TOWN3_CAR1",
                    "manual_control": False,
                    "auto_control": False,
                    "camera_type": "rgb",
                    "collision_sensor": "on",
                    "lane_sensor": "on",
                    "log_images": False,
                    "log_measurements": False,
                    "render": True,
                    "x_res": 168,
                    "y_res": 168,
                    "use_depth_camera": False,
                    "send_measurements": False,
                },
                "car2": {
                    "type": "vehicle_4W",
                    "enable_planner": True,
                    "convert_images_to_video": False,
                    "early_terminate_on_collision": True,
                    "reward_function": "corl2017",
                    "scenarios": "SSUI3C_TOWN3_CAR2",
                    "manual_control": False,
                    "auto_control": False,
                    "camera_type": "rgb",
                    "collision_sensor": "on",
                    "lane_sensor": "on",
                    "log_images": False,
                    "log_measurements": False,
                    "render": True,
                    "x_res": 168,
                    "y_res": 168,
                    "use_depth_camera": False,
                    "send_measurements": False,
                },
                "car3": {
                    "type": "vehicle_4W",
                    "enable_planner": True,
                    "convert_images_to_video": False,
                    "early_terminate_on_collision": True,
                    "reward_function": "corl2017",
                    "scenarios": "SSUI3C_TOWN3_CAR3",
                    "manual_control": False,
                    "auto_control": False,
                    "camera_type": "rgb",
                    "collision_sensor": "on",
                    "lane_sensor": "on",
                    "log_images": False,
                    "log_measurements": False,
                    "render": True,
                    "x_res": 168,
                    "y_res": 168,
                    "use_depth_camera": False,
                    "send_measurements": False,
                },
            },
        }
        super(StopSign3CarTown03, self).__init__(self.configs)


if __name__ == "__main__":
    env = StopSign3CarTown03()
    configs = env.configs
    for ep in range(2):
        obs = env.reset()

        total_reward_dict = {}
        action_dict = {}

        env_config = configs["env"]
        actor_configs = configs["actors"]
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
            # action_dict = get_next_actions(info, env.discrete_actions)
            for actor_id in total_reward_dict.keys():
                total_reward_dict[actor_id] += reward[actor_id]
            print(":{}\n\t".join(["Step#", "rew", "ep_rew",
                                  "done{}"]).format(i, reward,
                                                    total_reward_dict, done))

            time.sleep(0.1)

        print("{} fps".format(i / (time.time() - start)))
