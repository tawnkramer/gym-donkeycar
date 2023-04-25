"""Top-level package for OpenAI Gym Environments for Donkey Car."""
import os

from gym.envs.registration import register

from gym_donkeycar.envs.donkey_env import (
    AvcSparkfunEnv,
    CircuitLaunchEnv,
    GeneratedRoadsEnv,
    GeneratedTrackEnv,
    MiniMonacoEnv,
    MountainTrackEnv,
    RoboRacingLeagueTrackEnv,
    ThunderhillTrackEnv,
    WarehouseEnv,
    WarrenTrackEnv,
    WaveshareEnv,
)

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file) as file_handler:
    __version__ = file_handler.read().strip()

__author__ = """Tawn Kramer"""
__email__ = "tawnkramer@gmail.com"
__version__ = __version__

register(id="donkey-generated-roads-v0", entry_point="gym_donkeycar.envs.donkey_env:GeneratedRoadsEnv")

register(id="donkey-warehouse-v0", entry_point="gym_donkeycar.envs.donkey_env:WarehouseEnv")

register(id="donkey-avc-sparkfun-v0", entry_point="gym_donkeycar.envs.donkey_env:AvcSparkfunEnv")

register(id="donkey-generated-track-v0", entry_point="gym_donkeycar.envs.donkey_env:GeneratedTrackEnv")

register(id="donkey-mountain-track-v0", entry_point="gym_donkeycar.envs.donkey_env:MountainTrackEnv")

register(id="donkey-roboracingleague-track-v0", entry_point="gym_donkeycar.envs.donkey_env:RoboRacingLeagueTrackEnv")

register(id="donkey-waveshare-v0", entry_point="gym_donkeycar.envs.donkey_env:WaveshareEnv")

register(id="donkey-minimonaco-track-v0", entry_point="gym_donkeycar.envs.donkey_env:MiniMonacoEnv")

register(id="donkey-warren-track-v0", entry_point="gym_donkeycar.envs.donkey_env:WarrenTrackEnv")

register(id="donkey-thunderhill-track-v0", entry_point="gym_donkeycar.envs.donkey_env:ThunderhillTrackEnv")

register(id="donkey-circuit-launch-track-v0", entry_point="gym_donkeycar.envs.donkey_env:CircuitLaunchEnv")

__all__ = [
    "AvcSparkfunEnv",
    "CircuitLaunchEnv",
    "GeneratedRoadsEnv",
    "GeneratedTrackEnv",
    "MiniMonacoEnv",
    "MountainTrackEnv",
    "RoboRacingLeagueTrackEnv",
    "ThunderhillTrackEnv",
    "WarehouseEnv",
    "WarrenTrackEnv",
    "WaveshareEnv",
]
