from gym.spaces import Box
from env.carla.multi_env import MultiCarlaEnv
from env.carla.multi_env import DEFAULT_MULTIENV_CONFIG
import ray
from ray.rllib.evaluation.policy_evaluator import PolicyEvaluator
from ray.rllib.evaluation.metrics import collect_metrics
from ray.rllib.agents.pg.pg_policy_graph import PGPolicyGraph

# TODO: Add test specific MultiEnv configs
# TODO: Tests should run with render=False
BATCH_STEPS = 200
NUM_ENVS = 6
BATCH_MODE = "complete_episodes"

# TODO: Add test to make sure DEFAULT_MULTIENV_CONFIG is valid
configs = DEFAULT_MULTIENV_CONFIG
configs["env"]["render"] = False
configs["env"]["discrete_actions"] = False


def init():
    ray.init(num_cpus=16, num_gpus=2)


def test_rllib_batch_policy_eval(init_done=False):
    if not init_done:
        init()
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
        batch_mode=BATCH_MODE,
        batch_steps=BATCH_STEPS,
        num_envs=NUM_ENVS)
    for _ in range(NUM_ENVS):
        samples, count = evaluator.sample_with_count()
        # print("sample:", samples.policy_batches["def_policy"]["actions"])
        # count >= BATCH_STEPS for complete_episodes
        # == for truncate_episodes
        if BATCH_MODE == "complete_episodes":
            assert count >= BATCH_STEPS, "Expected count:{}. actual:{}".format(
                BATCH_STEPS, count)
        elif BATCH_MODE == "truncate_episodes":
            assert count == BATCH_STEPS, "Expected count:{}. actual:{}".format(
                BATCH_STEPS, count)
        print("Successfully sampled {} items".format(count))
    results = collect_metrics(evaluator, [])
    print("results: \n", results)
    if BATCH_MODE == "complete_episodes":
        assert (results["episodes"] >= NUM_ENVS), "Expected num episodes:{}," \
         "actual:{}".format(NUM_ENVS, results["episodes"])


if __name__ == "__main__":
    init()
    test_rllib_batch_policy_eval(init_done=True)
