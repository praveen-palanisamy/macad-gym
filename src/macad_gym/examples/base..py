import argparse
import time

from macad_gym.carla.multi_env import MultiActorCarlaEnv



def get_next_actions(measurements, is_discrete_actions):
    """Get/Update next action, work with way_point based planner.

    Args:
        measurements (dict): measurement data.
        is_discrete_actions (bool): whether to use discrete actions

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

# TODO fai pytest xml
#  pydoc with typed variables
#  solve eventual to dos
#  reorder the classes
#  check what is used and what is not
#  registered env scenario management
#  pettingzoo api             <-- implement AECEnv API
#  vec_env/parallel_env       <-- make example macad_gym.env(... a="", b="", ...)
#  test agents/rlib support   <-- https://github1s.com/LucasAlegre/sumo-rl/blob/HEAD/experiments/a3c_4x4grid.py#L17


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="CARLA Manual Control Client")
    # TODO: Fix the default path to the config.json;Should work after packaging
    argparser.add_argument("--xml_config_path", default="src/macad_gym/carla/configs.xml", help="Path to the xml config file")
    argparser.add_argument("--maps_path", default="/Game/Carla/Maps/", help="Path to the CARLA maps")
    argparser.add_argument("--render_mode", default="human", help="Path to the CARLA maps")

    args = vars(argparser.parse_args())
    env = MultiActorCarlaEnv(**args)

    for _ in range(2):
        obs = env.reset()

        total_reward_dict = {}
        action_dict = {}

        for actor_id in env.actor_configs.keys():
            total_reward_dict[actor_id] = 0
            if env.discrete_action_space:
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
            action_dict = get_next_actions(info, env.discrete_action_space)
            for actor_id in total_reward_dict.keys():
                total_reward_dict[actor_id] += reward[actor_id]
            print(
                ":{}\n\t".join(["Step#", "rew", "ep_rew", "done{}"]).format(
                    i, reward, total_reward_dict, done
                )
            )

        print("{} fps".format(i / (time.time() - start)))
