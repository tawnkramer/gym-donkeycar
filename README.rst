======================================================
OpenAI Gym Environments for Donkey Car
======================================================


.. image:: https://img.shields.io/pypi/v/gym-donkeycar.svg
        :target: https://pypi.python.org/pypi/gym-donkeycar

.. image:: https://img.shields.io/travis/leigh-johnson/gym-donkeycar.svg
        :target: https://travis-ci.org/leigh-johnson/gym-donkeycar

.. image:: https://readthedocs.org/projects/gym-donkeycar/badge/?version=latest
        :target: https://gym-donkeycar.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

Donkey Car OpenAI Gym


* Free software: MIT license
* Documentation: https://gym-donkeycar.readthedocs.io/en/latest/

Installation
--------------

* Download simulator binaries: https://github.com/tawnkramer/gym-donkeycar/releases


Example Usage
--------------

A short and compact introduction for people who know gym environments, but want to understand this one. 
Simple example code:

.. code-block:: python

    import os
    import gym
    import gym_donkeycar
    import numpy as np

    #%% SET UP ENVIRONMENT
    exe_path = f"{PATH_TO_APP}/donkey_sim.exe"
    port = 9091    
    
    conf = { "exe_path" : exe_path, "port" : port }

    env = gym.make("donkey-generated-track-v0", conf=conf)

    #%% PLAY
    obv = env.reset()
    for t in range(100):
        action = np.array([0.0,0.5]) # drive straight with small speed
    # execute the action
    obv, reward, done, info = env.step(action)



* see more examples: https://github.com/tawnkramer/gym-donkeycar/tree/master/examples

Action space
--------------

A permissable action is a numpy array of length two with first steering and throttle, respectively. E.g. np.array([0,1]) goes straight at full speed, np.array([-5,1]) turns left etc.

Action Space: Box(2,)

Action names: ['steer', 'throttle']


What you receive back on step
--------------

- obv: The image that the donkey is seeing (np.array shape (120,160,3))
- reward: a reward that combines game over, how far from center and speed (max=1, min approx -2)
- done: Boolean. Game over if cte > max_cte or hit != "none"

- info contains:
  - cte: Cross track error (how far from center line)
  - positions: x,y,z
  - speed: positive forward, negative backward
  - hit: 'none' if all is good.

example info:

.. code-block:: python

    {'pos': (51.49209, 0.7399381, 117.3004),
     'cte': -5.865292,
     'speed': 9.319956,
     'hit': 'none'}


Environments
---------------

* "donkey-warehouse-v0"
* "donkey-generated-roads-v0"
* "donkey-avc-sparkfun-v0"
* "donkey-generated-track-v0"
* "donkey-roboracingleague-track-v0"
* "donkey-waveshare-v0"
* "donkey-minimonaco-track-v0"
* "donkey-warren-track-v0"


Credits
------------

Original Source Code

Tawn Kramer - https://github.com/tawnkramer/gym-donkeycar

Roma Sokolkov - https://github.com/r7vme/gym-donkeycar cloned with permission from https://github.com/tawnkramer/sdsandbox

Maintainer

Maxime Ellerbach - https://github.com/Maximellerbach/gym-donkeycar

Release Engineer

.. _Leigh Johnson: https://github.com/leigh-johnson

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
