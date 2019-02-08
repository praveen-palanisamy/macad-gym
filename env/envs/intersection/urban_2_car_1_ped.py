#!/usr/bin/env python
import json
import time

from env.carla.multi_env import MultiCarlaEnv

config_file = open("urban_2_car_1_ped.json")
configs = json.load(config_file)
env = MultiCarlaEnv(configs)


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


for ep in range(2):
    obs = env.reset()
    for _ in range(2):
        obs = env.reset()
        total_vehicle = env.num_vehicle

        total_reward_dict = {}
        action_dict = {}

        env_config = configs["env"]
        actor_configs = configs["actors"]
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
            # action_dict = get_next_actions(info, env.discrete_actions)
            for actor_id in total_reward_dict.keys():
                total_reward_dict[actor_id] += reward[actor_id]
            print(":{}\n\t".join(["Step#", "rew", "ep_rew", "done{}"]).format(
                i, reward, total_reward_dict, done))

            time.sleep(0.1)

        print("{} fps".format(i / (time.time() - start)))

# Clean actors in world
env.clean_world()
