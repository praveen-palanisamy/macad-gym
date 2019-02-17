import argparse
import json
import math
import datetime

from tensorboardX import SummaryWriter

from env.carla.multi_env import MultiCarlaEnv, DEFAULT_MULTIENV_CONFIG, \
    DISCRETE_ACTIONS
from env.carla.agents.navigation.basic_agent import BasicAgent

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
writer = SummaryWriter("logs/" + timestamp)


def vehicle_control_to_action(vehicle_control, is_discrete):
    if vehicle_control.hand_brake:
        continuous_action = [-1.0, vehicle_control.steer]
    else:
        if vehicle_control.reverse:
            continuous_action = [
                vehicle_control.brake - vehicle_control.throttle,
                vehicle_control.steer
            ]
        else:
            continuous_action = [
                vehicle_control.throttle - vehicle_control.brake,
                vehicle_control.steer
            ]

    def dist(a, b):
        return math.sqrt((a[0] - b[0]) * (a[0] - b[0]) +
                         (a[1] - b[1]) * (a[1] - b[1]))

    if is_discrete:
        closest_action = 0
        shortest_action_distance = dist(continuous_action, DISCRETE_ACTIONS[0])

        for i in range(1, len(DISCRETE_ACTIONS)):
            d = dist(continuous_action, DISCRETE_ACTIONS[i])
            if d < shortest_action_distance:
                closest_action = i
                shortest_action_distance = d
        return closest_action

    return continuous_action


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='CARLA Basic Navigation')

    argparser.add_argument(
        '--config',
        # default='env/carla/config.json',
        help='specify json configuration file for environment setup')

    args = argparser.parse_args()

    if args.config is not None:
        multi_env_config = json.load(open(args.config))
    else:
        multi_env_config = DEFAULT_MULTIENV_CONFIG
    multi_env_config["env"]["enable_planner"] = True
    multi_env_config["env"]["discrete_actions"] = False
    env = MultiCarlaEnv(multi_env_config)

    env_config = multi_env_config["env"]
    actor_configs = multi_env_config["actors"]

    step = 0
    vehicle_dict = {}

    for ep in range(10):
        agent_dict = {}
        obs = env.reset()
        total_reward_dict = {k: 0.0 for k in actor_configs.keys()}
        for actor_id in actor_configs.keys():
            vehicle_dict[actor_id] = env.actors[actor_id]
            agent = BasicAgent(env.actors[actor_id], target_speed=40)
            agent.set_destination(env.end_pos[actor_id])
            agent_dict[actor_id] = agent

        done = False
        while not done:
            action_dict = {}
            for actor_id, agent in agent_dict.items():
                action_dict[actor_id] = vehicle_control_to_action(
                    agent.run_step(), env.discrete_actions)

            obs_dict, reward_dict, done_dict, info_dict = env.step(action_dict)
            done = done_dict["__all__"]

            for actor_id, rew in reward_dict.items():
                total_reward_dict[actor_id] += rew
                print("Fwd speed of", actor_id, ":",
                      info_dict[actor_id]["forward_speed"])
            print(":{}\n\t".join(["Step#", "rew", "ep_rew", "done{}"]).format(
                step, reward_dict, total_reward_dict, done_dict))
            step += 1
        for actor_id, ep_rew in total_reward_dict.items():
            writer.add_scalar(actor_id, ep_rew, step)
    # Clean actors in world
    env.clean_world()
    # env.camera_list.save_images_to_disk()

writer.close()
