"""Simple deterministic agent acting in the env loop."""

import argparse
import time

from carla_gym.multi_env import MultiActorCarlaEnv


def get_next_actions(measurements, is_discrete_actions):
    """Get/Update next action, work with way_point based planner.

    Args:
        measurements (dict): measurement data.
        is_discrete_actions (bool): whether to use discrete actions

    Returns:
        dict: actions, dict of len-two integer lists.
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
    argparser = argparse.ArgumentParser(description="CARLA Manual Control Client")
    argparser.add_argument("--xml_config_path", default="configs.xml", help="Path to the xml config file")
    argparser.add_argument("--maps_path", default="/Game/Carla/Maps/", help="Path to the CARLA maps")
    argparser.add_argument("--render_mode", default="none", help="Path to the CARLA maps")

    args = vars(argparser.parse_args())
    env = MultiActorCarlaEnv(**args)
    # otherwise: env = gym.make("carla-v0", **args)

    for _ in range(2):
        obs = env.reset()
        total_reward_dict = {}

        # agents init
        action_dict = {}
        for actor_id in env.actor_configs.keys():
            total_reward_dict[actor_id] = 0
            if env.discrete_action_space:
                action_dict[actor_id] = 3  # Forward
            else:
                action_dict[actor_id] = [1, 0]  # test values

        start = time.time()
        step = 0
        done = {"__all__": False}
        while not done["__all__"]:
            step += 1
            obs, reward, done, info = env.step(action_dict)
            action_dict = get_next_actions(info, env.discrete_action_space)
            for actor_id in total_reward_dict.keys():
                total_reward_dict[actor_id] += reward[actor_id]
            print(":{}\n\t".join(["Step#", "rew", "ep_rew", "done{}"]).format(step, reward, total_reward_dict, done))
        print(f"{step / (time.time() - start)} fps")
    env.close()
