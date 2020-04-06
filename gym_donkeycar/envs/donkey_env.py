'''
file: donkey_env.py
author: Tawn Kramer
date: 2018-08-31
'''
import os
import random
import time

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from gym_donkeycar.envs.donkey_sim import DonkeyUnitySimContoller
from gym_donkeycar.envs.donkey_proc import DonkeyUnityProcess


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
    THROTTLE_MAX = 5.0
    VAL_PER_PIXEL = 255

    def __init__(self, level=0, exe_path="self_start", host='127.0.0.1', port=9091, frame_skip=2, start_delay=5.0):

        print("starting DonkeyGym env")
        self.viewer = None

        # start Unity simulation subprocess
        self.proc = DonkeyUnityProcess()

        # the unity sim server will bind to the host ip given
        self.proc.start(exe_path, host='0.0.0.0', port=port)

        # wait for simulator to startup and begin listening
        time.sleep(start_delay)

        # start simulation com
        self.viewer = DonkeyUnitySimContoller(level=level, host=host, port=port)

        # steering and throttle
        self.action_space = spaces.Box(low=np.array([self.STEER_LIMIT_LEFT, self.THROTTLE_MIN]),
                                       high=np.array([self.STEER_LIMIT_RIGHT, self.THROTTLE_MAX]), dtype=np.float32)

        # camera sensor data
        self.observation_space = spaces.Box(
            0, self.VAL_PER_PIXEL, self.viewer.get_sensor_size(), dtype=np.uint8)

        # simulation related variables.
        self.seed()

        # Frame Skipping
        self.frame_skip = frame_skip

        # wait until loaded
        self.viewer.wait_until_loaded()

    def __del__(self):
        self.close()

    def close(self):
        if self.viewer:
            self.viewer.quit()
        self.proc.quit()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        for i in range(self.frame_skip):
            self.viewer.take_action(action)
            observation, reward, done, info = self.viewer.observe()
        return observation, reward, done, info

    def reset(self):
        self.viewer.reset()
        observation, reward, done, info = self.viewer.observe()
        time.sleep(1)
        return observation

    def render(self, mode="human", close=False):
        if close:
            self.viewer.quit()

        return self.viewer.render(mode)

    def is_game_over(self):
        return self.viewer.is_game_over()


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

class GeneratedRoadsEnv(DonkeyEnv):

    def __init__(self, *args, **kwargs):
        super(GeneratedRoadsEnv, self).__init__(level=0, *args, **kwargs)


class WarehouseEnv(DonkeyEnv):

    def __init__(self, *args, **kwargs):
        super(WarehouseEnv, self).__init__(level=1, *args, **kwargs)


class AvcSparkfunEnv(DonkeyEnv):

    def __init__(self, *args, **kwargs):
        super(AvcSparkfunEnv, self).__init__(level=2, *args, **kwargs)


class GeneratedTrackEnv(DonkeyEnv):

    def __init__(self, *args, **kwargs):
        super(GeneratedTrackEnv, self).__init__(level=3, *args, **kwargs)
