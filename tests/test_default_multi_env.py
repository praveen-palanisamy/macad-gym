"""Unit tests for each method in MultiCarlaEnv"""
from copy import deepcopy

from macad_gym.carla.multi_env import MultiCarlaEnv
from macad_gym.carla.multi_env import DEFAULT_MULTIENV_CONFIG
import gym.spaces
import pytest


@pytest.fixture
def make_default_env():
    configs = DEFAULT_MULTIENV_CONFIG
    configs["env"]["render"] = False
    configs["env"]["discrete_actions"] = False
    env = MultiCarlaEnv(configs)
    return env, configs


def test_multienv_traincycle():
    base_config = deepcopy(DEFAULT_MULTIENV_CONFIG)
    env = MultiCarlaEnv(base_config, args={"maps_path": "/Game/Carla/Maps/"})
    env.reset()

def test_multienv_argments():
    base_config = deepcopy(DEFAULT_MULTIENV_CONFIG)
    ok_noEnv_config = deepcopy(DEFAULT_MULTIENV_CONFIG)
    del ok_noEnv_config["env"]
    err_noActors_config = deepcopy(DEFAULT_MULTIENV_CONFIG)
    del err_noActors_config["actors"]
    err_duplicateActorKey_config = deepcopy(DEFAULT_MULTIENV_CONFIG)
    k,v = list(list(err_duplicateActorKey_config["scenarios"].values())[0]["ego_vehicles"].items())[0]
    list(err_duplicateActorKey_config["scenarios"].values())[0]["other_vehicles"].update({k: v})
    err_mismatchActorsKey_config = deepcopy(DEFAULT_MULTIENV_CONFIG)
    err_mismatchActorsKey_config["actors"].update({"err_missing_actor_key": list(err_mismatchActorsKey_config["actors"].values())[0]})
    del list(err_mismatchActorsKey_config["actors"].values())[0]


    env = MultiCarlaEnv(base_config)
    env = MultiCarlaEnv(ok_noEnv_config)
    # At least one scenario
    with pytest.raises(AssertionError):
        env = MultiCarlaEnv({})
        return False
    # Agents config should match
    with pytest.raises(ValueError):
        env = MultiCarlaEnv(err_noActors_config)
        return False
    # Agents config should match
    with pytest.raises(ValueError):
        env = MultiCarlaEnv(err_duplicateActorKey_config)
        return False
    # Agents config should match
    with pytest.raises(ValueError):
        env = MultiCarlaEnv(err_mismatchActorsKey_config)
        return False

    return True

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
                                         "Dict Obs space, one key-value pair"
                                         "per actor/agent")
