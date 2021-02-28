#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `gym_donkeycar.core` package."""

from gym_donkeycar.core.fps import FPSTimer

# @pytest.fixture
# def response():
#     """Sample pytest fixture.

#     See more at: http://doc.pytest.org/en/latest/fixture.html
#     """
#     # import requests
#     # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_fps():
    """
    it should increment iter on_frame
    it should reset t and iter
    """

    timer = FPSTimer()

    assert timer.iter == 0

    timer.on_frame()

    assert timer.iter == 1

    timer.reset()

    assert timer.iter == 0
