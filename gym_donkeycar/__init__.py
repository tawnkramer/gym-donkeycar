# -*- coding: utf-8 -*-

"""Top-level package for OpenAI Gym Environments for Donkey Car."""

__author__ = """Tawn Kramer"""
__email__ = 'tawnkramer@gmail.com'
__version__ = '1.0.14'

from gym.envs.registration import register
from .envs.donkey_env import GeneratedRoadsEnv, WarehouseEnv, AvcSparkfunEnv, GeneratedTrackEnv, MountainTrackEnv

register(
    id='donkey-generated-roads-v0',
    entry_point='gym_donkeycar.envs.donkey_env:GeneratedRoadsEnv',
)

register(
    id='donkey-warehouse-v0',
    entry_point='gym_donkeycar.envs.donkey_env:WarehouseEnv',
)

register(
    id='donkey-avc-sparkfun-v0',
    entry_point='gym_donkeycar.envs.donkey_env:AvcSparkfunEnv',
)

register(
    id='donkey-generated-track-v0',
    entry_point='gym_donkeycar.envs.donkey_env:GeneratedTrackEnv',
)

register(
    id='donkey-mountain-track-v0',
    entry_point='gym_donkeycar.envs.donkey_env:MountainTrackEnv',
)
