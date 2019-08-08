#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `gym_donkeycar` package."""

import os
import pytest

import gym
import gym_donkeycar.envs

env_list = [
       "donkey-warehouse-v0",
       "donkey-generated-roads-v0",
       "donkey-avc-sparkfun-v0",
       "donkey-generated-track-v0"
]

def test_load_gyms(mocker):
    sim_ctl = mocker.patch('gym_donkeycar.envs.donkey_env.DonkeyUnitySimContoller')
    unity_proc = mocker.patch('gym_donkeycar.envs.donkey_env.DonkeyUnityProcess')

    for i, gym_name in enumerate(env_list):

        env = gym.make(gym_name)
        assert env.ACTION_NAMES == ['steer', 'throttle']
        assert env.spec.id == gym_name
        assert sim_ctl.call_count == i+1
        assert unity_proc.call_count == i+1

