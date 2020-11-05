'''
file: donkey_env.py
author: Tawn Kramer
date: 2018-08-31
'''
import os
import random
import time
import logging

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from gym_donkeycar.envs.donkey_sim import DonkeyUnitySimContoller
from gym_donkeycar.envs.donkey_proc import DonkeyUnityProcess

logger = logging.getLogger(__name__)

def supply_defaults(conf):
    """
    Supply all defaults.

    Args:
        conf: (todo): write your description
    """
    defaults = [("start_delay", 5.0),
                ("max_cte", 5.0),
                ("frame_skip", 2),
                ("cam_resolution", (120,160,3)),
                ("log_level", logging.INFO)]

    for key, val in defaults:
        if not key in conf:
            conf[key] = val
            print("setting default: %s %s" % (key, val.__str__()))


class DonkeyEnv(gym.Env):
    """
    OpenAI Gym Environment for Donkey
    """

    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    ACTION_NAMES = ["steer", "throttle"]
    STEER_LIMIT_LEFT = -1.0
    STEER_LIMIT_RIGHT = 1.0
    THROTTLE_MIN = 0.0
    THROTTLE_MAX = 1.0
    VAL_PER_PIXEL = 255

    def __init__(self, level, conf):
        """
        Initialize the sensor

        Args:
            self: (todo): write your description
            level: (int): write your description
            conf: (todo): write your description
        """
        print("starting DonkeyGym env")
        self.viewer = None
        self.proc = None
        conf["level"] = level

        # ensure defaults are supplied if missing.
        supply_defaults(conf)        

        # set logging level
        logging.basicConfig(level=conf["log_level"])

        logger.debug("DEBUG ON")
        logger.debug(conf)

        # start Unity simulation subprocess
        self.proc = DonkeyUnityProcess()

        # the unity sim server will bind to the host ip given
        self.proc.start(conf['exe_path'], host='0.0.0.0', port=conf['port'])

        # wait for simulator to startup and begin listening
        time.sleep(conf["start_delay"])

        # start simulation com
        self.viewer = DonkeyUnitySimContoller(conf=conf)

        # steering and throttle
        self.action_space = spaces.Box(low=np.array([self.STEER_LIMIT_LEFT, self.THROTTLE_MIN]),
                                       high=np.array([self.STEER_LIMIT_RIGHT, self.THROTTLE_MAX]), dtype=np.float32)

        # camera sensor data
        self.observation_space = spaces.Box(
            0, self.VAL_PER_PIXEL, self.viewer.get_sensor_size(), dtype=np.uint8)

        # simulation related variables.
        self.seed()

        # Frame Skipping
        self.frame_skip = conf["frame_skip"]

        # wait until loaded
        self.viewer.wait_until_loaded()


    def __del__(self):
        """
        Closes the stream.

        Args:
            self: (todo): write your description
        """
        self.close()

    def close(self):
        """
        Closes the stream.

        Args:
            self: (todo): write your description
        """
        if hasattr(self, "viewer") and self.viewer is not None:
            self.viewer.quit()
        if hasattr(self, "proc") and self.proc is not None:
            self.proc.quit()

    def set_reward_fn(self, reward_fn):
        """
        Sets the reward function.

        Args:
            self: (todo): write your description
            reward_fn: (todo): write your description
        """
        self.viewer.set_reward_fn(reward_fn)

    def set_episode_over_fn(self, ep_over_fn):
        """
        Set the episode function.

        Args:
            self: (todo): write your description
            ep_over_fn: (bool): write your description
        """
        self.viewer.set_episode_over_fn(ep_over_fn)

    def seed(self, seed=None):
        """
        Return a random seed.

        Args:
            self: (todo): write your description
            seed: (int): write your description
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        Perform a single step

        Args:
            self: (todo): write your description
            action: (int): write your description
        """
        for i in range(self.frame_skip):
            self.viewer.take_action(action)
            observation, reward, done, info = self.viewer.observe()
        return observation, reward, done, info

    def reset(self):
        """
        Reset the observation.

        Args:
            self: (todo): write your description
        """
        self.viewer.reset()
        observation, reward, done, info = self.viewer.observe()
        time.sleep(1)
        return observation

    def render(self, mode="human", close=False):
        """
        Render the viewer.

        Args:
            self: (todo): write your description
            mode: (str): write your description
            close: (bool): write your description
        """
        if close:
            self.viewer.quit()

        return self.viewer.render(mode)

    def is_game_over(self):
        """
        Return true if the game is over a game.

        Args:
            self: (todo): write your description
        """
        return self.viewer.is_game_over()


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

class GeneratedRoadsEnv(DonkeyEnv):

    def __init__(self, *args, **kwargs):
        """
        Initialize the class.

        Args:
            self: (todo): write your description
        """
        super(GeneratedRoadsEnv, self).__init__(level='generated_road', *args, **kwargs)


class WarehouseEnv(DonkeyEnv):

    def __init__(self, *args, **kwargs):
        """
        Initialize the initial state.

        Args:
            self: (todo): write your description
        """
        super(WarehouseEnv, self).__init__(level='warehouse', *args, **kwargs)


class AvcSparkfunEnv(DonkeyEnv):

    def __init__(self, *args, **kwargs):
        """
        Initialize the module.

        Args:
            self: (todo): write your description
        """
        super(AvcSparkfunEnv, self).__init__(level='sparkfun_avc', *args, **kwargs)


class GeneratedTrackEnv(DonkeyEnv):

    def __init__(self, *args, **kwargs):
        """
        Initialize the class.

        Args:
            self: (todo): write your description
        """
        super(GeneratedTrackEnv, self).__init__(level='generated_track', *args, **kwargs)


class MountainTrackEnv(DonkeyEnv):

    def __init__(self, *args, **kwargs):
        """
        Initialize an initialiser

        Args:
            self: (todo): write your description
        """
        super(MountainTrackEnv, self).__init__(level='mountain_track', *args, **kwargs)


class RoboRacingLeagueTrackEnv(DonkeyEnv):

    def __init__(self, *args, **kwargs):
        """
        Initialize the greenlet

        Args:
            self: (todo): write your description
        """
        super(RoboRacingLeagueTrackEnv, self).__init__(level='roboracingleague_1', *args, **kwargs)
