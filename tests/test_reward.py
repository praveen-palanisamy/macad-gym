import carla_gym
from carla_api.reward import Reward, REWARD_CORL2017, REWARD_LANE_KEEP, REWARD_CUSTOM
from core.constants import DEFAULT_MULTIENV_CONFIG

MAPS_PATH = "/Game/Carla/Maps/"


def test_reward_policies():
    """Test the available reward policies."""
    env = carla_gym.env(configs=DEFAULT_MULTIENV_CONFIG, maps_path=MAPS_PATH)
    env.reset()
    env.step(0)
    info = env.last()[-1]

    reward_policy = Reward()
    reward_policy.compute_reward(info, info, REWARD_CORL2017)
    reward_policy.compute_reward(info, info, REWARD_LANE_KEEP)
    reward_policy.compute_reward(info, info, REWARD_CUSTOM)

    env.close()
