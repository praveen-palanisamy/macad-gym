"""Unit tests for each method in MultiCarlaEnv"""
from macad_gym.carla.multi_env import MultiCarlaEnv
from macad_gym.carla.multi_env import DEFAULT_MULTIENV_CONFIG
import gym.spaces

# TODO: Add test to make sure DEFAULT_MULTIENV_CONFIG is valid
configs = DEFAULT_MULTIENV_CONFIG
configs["env"]["render"] = False
configs["env"]["discrete_actions"] = False


def test_multienv_action_space():
    env = MultiCarlaEnv(configs)
    assert isinstance(env.action_space,
                      gym.spaces.Dict), ("Multi Actor/Agent environment should"
                                         "have Dict action space, one"
                                         "key-value pair per actor/agent")
