"""
file: donkey_env.py
author: Tawn Kramer
date: 2018-08-31
"""
import logging
import time

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from gym_donkeycar.envs.donkey_proc import DonkeyUnityProcess
from gym_donkeycar.envs.donkey_sim import DonkeyUnitySimContoller

logger = logging.getLogger(__name__)


def supply_defaults(conf):
    defaults = [
        ("start_delay", 5.0),
        ("max_cte", 5.0),
        ("frame_skip", 2),
        ("cam_resolution", (120, 160, 3)),
        ("log_level", logging.INFO),
        ("host", "localhost"),
        ("port", 9091),
    ]

    for key, val in defaults:
        if key not in conf:
            conf[key] = val
            print(f"Setting default: {key} {val}")


class DonkeyEnv(gym.Env):
    """
    OpenAI Gym Environment for Donkey
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    ACTION_NAMES = ["steer", "throttle"]
    STEER_LIMIT_LEFT = -1.0
    STEER_LIMIT_RIGHT = 1.0
    THROTTLE_MIN = 0.0
    THROTTLE_MAX = 1.0
    VAL_PER_PIXEL = 255

    def __init__(self, level, conf=None):
        print("starting DonkeyGym env")
        self.viewer = None
        self.proc = None

        if conf is None:
            conf = {}

        conf["level"] = level

        # ensure defaults are supplied if missing.
        supply_defaults(conf)

        # set logging level
        logging.basicConfig(level=conf["log_level"])  # pytype: disable=key-error

        logger.debug("DEBUG ON")
        logger.debug(conf)

        # start Unity simulation subprocess
        self.proc = None
        if "exe_path" in conf:
            self.proc = DonkeyUnityProcess()
            # the unity sim server will bind to the host ip given
            self.proc.start(conf["exe_path"], host="0.0.0.0", port=conf["port"])

            # wait for simulator to startup and begin listening
            time.sleep(conf["start_delay"])

        # start simulation com
        self.viewer = DonkeyUnitySimContoller(conf=conf)

        # steering and throttle
        self.action_space = spaces.Box(
            low=np.array([self.STEER_LIMIT_LEFT, self.THROTTLE_MIN]),
            high=np.array([self.STEER_LIMIT_RIGHT, self.THROTTLE_MAX]),
            dtype=np.float32,
        )

        # camera sensor data
        self.observation_space = spaces.Box(0, self.VAL_PER_PIXEL, self.viewer.get_sensor_size(), dtype=np.uint8)

        # simulation related variables.
        self.seed()

        # Frame Skipping
        self.frame_skip = conf["frame_skip"]  # pytype: disable=key-error

        # wait until loaded
        self.viewer.wait_until_loaded()

    def __del__(self):
        self.close()

    def close(self):
        if hasattr(self, "viewer") and self.viewer is not None:
            self.viewer.quit()
        if hasattr(self, "proc") and self.proc is not None:
            self.proc.quit()

    def set_reward_fn(self, reward_fn):
        self.viewer.set_reward_fn(reward_fn)

    def set_episode_over_fn(self, ep_over_fn):
        self.viewer.set_episode_over_fn(ep_over_fn)

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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class GeneratedRoadsEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super(GeneratedRoadsEnv, self).__init__(level="generated_road", *args, **kwargs)


class WarehouseEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super(WarehouseEnv, self).__init__(level="warehouse", *args, **kwargs)


class AvcSparkfunEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super(AvcSparkfunEnv, self).__init__(level="sparkfun_avc", *args, **kwargs)


class GeneratedTrackEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super(GeneratedTrackEnv, self).__init__(level="generated_track", *args, **kwargs)


class MountainTrackEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super(MountainTrackEnv, self).__init__(level="mountain_track", *args, **kwargs)


class RoboRacingLeagueTrackEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super(RoboRacingLeagueTrackEnv, self).__init__(level="roboracingleague_1", *args, **kwargs)


class WaveshareEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super(WaveshareEnv, self).__init__(level="waveshare", *args, **kwargs)


class MiniMonacoEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super(MiniMonacoEnv, self).__init__(level="mini_monaco", *args, **kwargs)


class WarrenTrackEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super(WarrenTrackEnv, self).__init__(level="warren", *args, **kwargs)


class ThunderhillTrackEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super(ThunderhillTrackEnv, self).__init__(level="thunderhill", *args, **kwargs)


class CircuitLaunchEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super(CircuitLaunchEnv, self).__init__(level="circuit_launch", *args, **kwargs)
