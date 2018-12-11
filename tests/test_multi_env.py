"""Unit tests for each method in MultiCarlaEnv"""

import gym
from env.carla.multi_env import DEFAULT_MULTIENV_CONFIG
from env.carla.multi_env import MultiCarlaEnv

def test_env_params():
    pass


def test_env_map_param():
    configs = DEFAULT_MULTIENV_CONFIG
    configs["server_map"] = 0
    env = MultiCarlaEnv(configs)
    ...


def test_action_space():
    configs = DEFAULT_MULTIENV_CONFIG
    configs["discrete_actions"] = True
    env = MultiCarlaEnv(configs)
    assert isinstance(env.action_space, gym.spaces.Discrete)

def test_invalid_action():
    configs = DEFAULT_MULTIENV_CONFIG
    configs["server_map"] = 0
    env = MultiCarlaEnv(configs)
    invalid_action = [0.1]