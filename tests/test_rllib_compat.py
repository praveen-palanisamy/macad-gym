from gym.spaces import Box
from macad_gym.carla.multi_env import MultiCarlaEnv
from macad_gym.carla.multi_env import DEFAULT_MULTIENV_CONFIG
import ray
from ray.rllib.evaluation.policy_evaluator import PolicyEvaluator
from ray.rllib.agents.pg.pg_policy_graph import PGPolicyGraph

# TODO: Add test specific MultiEnv configs
# TODO: Tests should run with render=False
BATCH_COUNT = 2
EPISODE_HORIZON = 10

# TODO: Add test to make sure DEFAULT_MULTIENV_CONFIG is valid
configs = DEFAULT_MULTIENV_CONFIG
configs["env"]["render"] = False
configs["env"]["discrete_actions"] = False


def init():
    ray.init(num_cpus=16, num_gpus=2)


def test_rllib_policy_eval(init_done=False):
    if not init_done:
        init()
    assert (
        not configs["env"]["render"]), "Tests should be run with render=False"
    evaluator = PolicyEvaluator(
        env_creator=lambda _: MultiCarlaEnv(configs),
        # TODO: Remove the hardcoded spaces
        policy_graph={
            "def_policy": (PGPolicyGraph, Box(0.0, 255.0, shape=(84, 84, 3)),
                           Box(-1.0, 1.0, shape=(2, )), {
                               "gamma": 0.99
                           })
        },
        policy_mapping_fn=lambda agent_id: "def_policy",
        batch_steps=BATCH_COUNT,
        episode_horizon=EPISODE_HORIZON)
    samples, count = evaluator.sample_with_count()
    print("Collected {} samples".format(count))
    assert count == BATCH_COUNT


if __name__ == "__main__":
    init()
    test_rllib_policy_eval(init_done=True)
