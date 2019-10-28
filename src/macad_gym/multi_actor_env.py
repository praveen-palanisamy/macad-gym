"""
multi_actor_env.py MACAD-Gym multi actor env interface
__author__:PP
"""

import gym


class MultiActorEnv(gym.Env):
    """An environment that hosts multiple independent actors.

    Actor are identified by actor ids(string).

    Examples:
        >>> env = MyMultiActorEnv()
        >>> obs = env.reset()
        >>> print(obs)
        {
            "car_0": [2.4, 1.6],
            "car_1": [3.4, -3.2],
            "camera_0": [0.0, 0.0, 10.0, 20.0. 30.0],
            "traffic_light_1": [0, 3, 5, 1],
        }
        >>> obs, rewards, dones, infos = env.step(
            action_dict={
            "car_0": 1, "car_1": 0, "camera_0": 1 "traffic_light_1": 2,
            })
        >>> print(rewards)
        {
            "car_0": 3,
            "car_1": -1,
            "camera_0": 1,
            "traffic_light_1": 0,
        }
        >>> print(dones)
        {
            "car_0": False,
            "car_1": True,
            "camera_0": False,
            "traffic_light_1": False,
            "__all__": False,
        }
    """

    _gym_disable_underscore_compat = True

    def reset(self):
        """Resets the env and returns observations from ready actors.

        Returns:
            obs (dict): New observations for each ready actors.
        """
        raise NotImplementedError

    def step(self, action_dict):
        """Returns observations from ready actors.

        The returns are dicts mapping from actor_id strings to values. The
        number of actors in the env can vary over time.

        Returns
        -------
            obs (dict): New observations for each ready actor.
            rewards (dict): Reward values for each ready actor. If the
                episode is just started, the value will be None.
            dones (dict): Done values for each ready actor. The special key
                "__all__" is used to indicate env termination.
            infos (dict): Info values for each ready actor.
        """
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def seed(self):
        raise NotImplementedError
