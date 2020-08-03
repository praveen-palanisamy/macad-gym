#!/bin/env python
import gym
import macad_gym  # noqa F401

env = gym.make("HomoNcomIndePOIntrxMASS3CTWN3-v0")
configs = env.configs
env_config = configs["env"]
actor_configs = configs["actors"]


class SimpleAgent(object):
    def __init__(self, actor_configs):
        """A simple, deterministic agent for an example
        Args:
            actor_configs: Actor config dict
        """
        self.actor_configs = actor_configs
        self.action_dict = {}

    def get_action(self, obs):
        """ Returns `action_dict` containing actions for each agent in the env
        """
        for actor_id in self.actor_configs.keys():
            # ... Process obs of each agent and generate action ...
            if env_config["discrete_actions"]:
                self.action_dict[actor_id] = 3  # Drive forward
            else:
                self.action_dict[actor_id] = [1, 0]  # Full-throttle
        return self.action_dict


agent = SimpleAgent(actor_configs)  # Plug-in your agent or use MACAD-Agents
for ep in range(2):
    obs = env.reset()
    done = {"__all__": False}
    step = 0
    while not done["__all__"]:
        obs, reward, done, info = env.step(agent.get_action(obs))
        print(f"Step#:{step}  Rew:{reward}  Done:{done}")
        step += 1
env.close()
