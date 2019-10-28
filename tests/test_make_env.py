import gym
import gym.spaces
import macad_gym  # noqa F401 ignore unused; Needed for Gym env registration


def test_homo_ncom_inde_po_intrx_mass3ctwn3_env():
    env = gym.make("HomoNcomIndePOIntrxMASS3CTWN3-v0")
    assert isinstance(env.action_space,
                      gym.spaces.Dict), ("Multi Actor/Agent environment should"
                                         "have Dict action space, one"
                                         "key-value pair per actor/agent")
    assert isinstance(env.observation_space,
                      gym.spaces.Dict), ("Multi Actor/Agent env should have "
                                         "Dict Obs space, one key-value pair"
                                         "per actor/agent")


def test_hete_ncom_inde_po_intrx_matls1b2c1ptwn3():
    env = gym.make("HeteNcomIndePOIntrxMATLS1B2C1PTWN3-v0")
    assert isinstance(env.action_space,
                      gym.spaces.Dict), ("Multi Actor/Agent environment should"
                                         "have Dict action space, one"
                                         "key-value pair per actor/agent")
    assert isinstance(env.observation_space,
                      gym.spaces.Dict), ("Multi Actor/Agent env should have "
                                         "Dict Obs space, one key-value pair"
                                         "per actor/agent")
