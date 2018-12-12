from gym.spaces import Box
from env.carla.multi_env import MultiCarlaEnv
from env.carla.multi_env import DEFAULT_MULTIENV_CONFIG
from ray.rllib.evaluation.policy_evaluator import PolicyEvaluator
from ray.rllib.agents.pg.pg_policy_graph import PGPolicyGraph

# TODO: Add test specific MultiEnv configs
# TODO: Tests should run with render=False
BATCH_COUNT = 2
EPISODE_HORIZON = 10

# TODO: Add test to make sure DEFAULT_MULTIENV_CONFIG is valid
configs = DEFAULT_MULTIENV_CONFIG
configs["env"]["render"] = False


def test_rllib_policy_eval():
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
    assert count == BATCH_COUNT


if __name__ == "__main__":
    test_rllib_policy_eval()
