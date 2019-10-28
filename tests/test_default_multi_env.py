"""Unit tests for each method in MultiCarlaEnv"""
from macad_gym.carla.multi_env import MultiCarlaEnv
from macad_gym.carla.multi_env import DEFAULT_MULTIENV_CONFIG
import gym.spaces
import pytest


@pytest.fixture
def make_default_env():
    # TODO: Add test to make sure DEFAULT_MULTIENV_CONFIG is valid
    configs = DEFAULT_MULTIENV_CONFIG
    configs["env"]["render"] = False
    configs["env"]["discrete_actions"] = False
    env = MultiCarlaEnv(configs)
    return env, configs


def test_multienv_action_space(make_default_env):
    env, configs = make_default_env
    assert isinstance(env.action_space,
                      gym.spaces.Dict), ("Multi Actor/Agent environment should"
                                         "have Dict action space, one"
                                         "key-value pair per actor/agent")


def test_multienv_obs_space(make_default_env):
    env, configs = make_default_env
    assert isinstance(env.observation_space,
                      gym.spaces.Dict), ("Multi Actor/Agent env should have "
                                         "Dict Obs space, one key-value paur"
                                         "per actor/agent")
