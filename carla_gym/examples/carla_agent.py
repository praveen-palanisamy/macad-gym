"""CARLA basic agent acting in the env loop."""

import argparse
import math
import time

from carla_gym.multi_env import MultiActorCarlaEnv, DISCRETE_ACTIONS
from carla_gym.carla_api.PythonAPI.agents.navigation.basic_agent import BasicAgent
from carla_gym.core.maps.nav_utils import get_next_waypoint


def vehicle_control_to_action(vehicle_control, is_discrete):
    """Vehicle control object to action."""
    if vehicle_control.hand_brake:
        continuous_action = [-1.0, vehicle_control.steer]
    else:
        if vehicle_control.reverse:
            continuous_action = [vehicle_control.brake - vehicle_control.throttle, vehicle_control.steer]
        else:
            continuous_action = [vehicle_control.throttle - vehicle_control.brake, vehicle_control.steer]

    def dist(a, b):
        """Distance function."""
        return math.sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]))

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
    argparser = argparse.ArgumentParser(description="CARLA Manual Control Client")
    argparser.add_argument("--xml_config_path", default="configs.xml", help="Path to the xml config file")
    argparser.add_argument("--maps_path", default="/Game/Carla/Maps/", help="Path to the CARLA maps")
    argparser.add_argument("--render_mode", default="human", help="Path to the CARLA maps")

    args = vars(argparser.parse_args())
    args["discrete_action_space"] = True
    # The scenario xml config should have "enable_planner" flag
    env = MultiActorCarlaEnv(**args)
    # otherwise: env = gym.make("carla-v0", **args)

    for _ in range(2):
        agent_dict = {}
        obs = env.reset()
        total_reward_dict = {}

        # agents init
        for actor_id in env.actor_configs.keys():
            # Set the goal for the planner to be 0.2 m after the destination just to be sure
            dest_loc = get_next_waypoint(env.world, env._end_pos[actor_id], 0.2)
            agent = BasicAgent(env._scenario_objects[actor_id], target_speed=40)
            agent.set_destination(dest_loc)
            agent_dict[actor_id] = agent

        start = time.time()
        step = 0
        done = {"__all__": False}
        while not done["__all__"]:
            step += 1
            action_dict = {}
            for actor_id, agent in agent_dict.items():
                action_dict[actor_id] = vehicle_control_to_action(agent.run_step(), env.discrete_action_space)
            obs, reward, done, info = env.step(action_dict)
            for actor_id in total_reward_dict.keys():
                total_reward_dict[actor_id] += reward[actor_id]
            print(":{}\n\t".join(["Step#", "rew", "ep_rew", "done{}"]).format(step, reward, total_reward_dict, done))
        print(f"{step / (time.time() - start)} fps")
    env.close()
