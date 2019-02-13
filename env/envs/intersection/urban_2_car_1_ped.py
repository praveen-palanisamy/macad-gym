#!/usr/bin/env python
import json
import time

from env.carla.multi_env import MultiCarlaEnv
# from env.carla.multi_env import get_next_actions

config_file = open("urban_2_car_1_ped.json")
configs = json.load(config_file)
env = MultiCarlaEnv(configs)

for ep in range(2):
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
