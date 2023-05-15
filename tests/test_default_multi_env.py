"""Unit tests for each method in MultiCarlaEnv"""
import importlib
import os
import random
import time
import warnings
from copy import deepcopy

import numpy as np
import pytest
from gymnasium.spaces import Dict

from pettingzoo.test import api_test

import carla_gym
from core.constants import DEFAULT_MULTIENV_CONFIG
from multi_env import MultiActorCarlaEnv

EXAMPLE_CONFIG_PATH = "../carla_gym/examples/configs.xml"


def parallel_api_test(env, num_cycles=1000):
    def sample_action(env, obs, agent):
        agent_obs = obs[agent]
        if isinstance(agent_obs, dict) and "action_mask" in agent_obs:
            legal_actions = np.flatnonzero(agent_obs["action_mask"])
            if len(legal_actions) == 0:
                return 0

            return random.choice(legal_actions)
        return env.action_space(agent).sample()

    # checks that reset takes arguments seed and options
    env.reset(seed=0, options={"options": 1})

    MAX_RESETS = 2
    for _ in range(MAX_RESETS):
        obs = env.reset()
        assert isinstance(obs, dict)
        assert set(obs.keys()) == (set(env.agents))
        terminated = {agent: False for agent in env.agents}
        truncated = {agent: False for agent in env.agents}
        live_agents = set(env.agents[:])
        has_finished = set()
        for _ in range(num_cycles):
            actions = {
                agent: sample_action(env, obs, agent)
                for agent in env.agents
                if ((agent in terminated and not terminated[agent]) or (agent in truncated and not truncated[agent]))
            }
            obs, rew, terminated, truncated, info = env.step(actions)
            for agent in env.agents:
                assert agent not in has_finished, "agent cannot be revived once dead"

                if agent not in live_agents:
                    live_agents.add(agent)

            assert isinstance(obs, dict)
            assert isinstance(rew, dict)
            assert isinstance(terminated, dict)
            assert isinstance(truncated, dict)
            assert isinstance(info, dict)

            agents_set = set(live_agents)
            keys = "observation reward terminated truncated info".split()
            vals = [obs, rew, terminated, truncated, info]
            for k, v in zip(keys, vals):
                key_set = set(v.keys())
                if key_set == agents_set:
                    continue
                if len(key_set) < len(agents_set):
                    warnings.warn(f"Live agent was not given {k}")
                else:
                    warnings.warn(f"Agent was given {k} but was dead last turn")

            if hasattr(env, "possible_agents"):
                assert set(env.agents).issubset(
                    set(env.possible_agents)
                ), "possible_agents defined but does not contain all agents"

                has_finished |= {
                    agent
                    for agent, d in [(x[0], x[1] or y[1]) for x, y in zip(terminated.items(), truncated.items())]
                    if d
                }
                if not env.agents and has_finished != set(env.possible_agents):
                    warnings.warn("No agents present but not all possible_agents are terminated or truncated")
            elif not env.agents:
                warnings.warn("No agents present")

            for agent in env.agents:
                assert env.observation_space(agent) is env.observation_space(agent), (
                    "observation_space should return the exact same space object (not a copy)"
                    "for an agent. Consider decorating your observation_space(self, agent) method"
                    "with @functools.lru_cache(maxsize=None)"
                )
                assert env.action_space(agent) is env.action_space(agent), (
                    "action_space should return the exact same space object (not a copy) for an agent"
                    "(ensures that action space seeding works as expected). Consider decorating your"
                    "action_space(self, agent) method with @functools.lru_cache(maxsize=None)"
                )

            for agent, d in [(x[0], x[1] or y[1]) for x, y in zip(terminated.items(), truncated.items())]:
                if d:
                    live_agents.remove(agent)

            assert set(env.agents) == live_agents, f"{env.agents} != {live_agents}"

            if len(live_agents) == 0:
                break


@pytest.fixture
def make_default_raw_env():
    env = MultiActorCarlaEnv(xml_config_path=EXAMPLE_CONFIG_PATH)
    return env


# Tests with raw env class and XML parser
def test_multienv_action_space(make_default_raw_env):
    env = make_default_raw_env
    assert isinstance(env.action_spaces, Dict), (
        "Multi Actor/Agent environment should" "have Dict action space, one" "key-value pair per actor/agent"
    )


def test_multienv_obs_space(make_default_raw_env):
    env = make_default_raw_env
    assert isinstance(env.observation_spaces, Dict), (
        "Multi Actor/Agent env should have " "Dict Obs space, one key-value pair" "per actor/agent"
    )


def test_env_traincycle(make_default_raw_env):
    env = make_default_raw_env

    # agents init
    action_dict = {}
    for actor_id in env.actor_configs.keys():
        if env.discrete_action_space:
            action_dict[actor_id] = 3  # Forward
        else:
            action_dict[actor_id] = [1, 0]  # test values

    env.reset()
    for _ in range(100):
        env.step(action_dict)
    env.close()
    time.sleep(5)


# Configs tests with JSON parser
def test_configs_behaviour():
    base_config = deepcopy(DEFAULT_MULTIENV_CONFIG)

    err_noScenarios_config = deepcopy(DEFAULT_MULTIENV_CONFIG)
    del err_noScenarios_config["scenarios"]
    noActors_config = deepcopy(DEFAULT_MULTIENV_CONFIG)
    del noActors_config["actors"]
    err_mismatchActorsKey_config = deepcopy(DEFAULT_MULTIENV_CONFIG)
    err_mismatchActorsKey_config["actors"].update(
        {"err_missing_actor_key": list(err_mismatchActorsKey_config["actors"].values())[0]}
    )

    # XML or JSON config should be provided
    with pytest.raises(AssertionError):
        env = MultiActorCarlaEnv(configs={})
        return False
    # At least one scenario
    with pytest.raises(AssertionError):
        env = MultiActorCarlaEnv(configs=err_noScenarios_config)
        return False
    # No actor is fine (e.g. only autopilot objects can be used)
    env = MultiActorCarlaEnv(configs=noActors_config)
    env.reset()
    for _ in range(10):
        env.step({})
    env.close()
    time.sleep(5)
    # Agents config should match
    with pytest.raises(ValueError):
        env = MultiActorCarlaEnv(configs=err_mismatchActorsKey_config)
        return False

    env = MultiActorCarlaEnv(configs=base_config)

    return True


# Test packed XML files
def test_exported_xml():
    with importlib.resources.path("carla_gym.scenarios", "") as package_path:
        for resource in os.listdir(package_path):
            if resource.endswith(".xml"):
                carla_gym.env(xml_config_path=package_path / resource)


# Petting Zoo tests
def test_petting_zoo_aec_api():
    env = carla_gym.env(xml_config_path=EXAMPLE_CONFIG_PATH, max_steps=500)
    api_test(env, num_cycles=1000, verbose_progress=True)
    env.close()
    time.sleep(5)


def test_petting_zoo_parallel_api():
    env = carla_gym.parallel_env(xml_config_path=EXAMPLE_CONFIG_PATH, max_steps=500)
    parallel_api_test(env, num_cycles=1000)
    env.close()
    time.sleep(5)
