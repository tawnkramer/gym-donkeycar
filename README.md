# OpenAI Gym Environments for Donkey Car

[![pypi](https://img.shields.io/pypi/v/gym-donkeycar.svg)](https://pypi.python.org/pypi/gym-donkeycar) [![CI](https://github.com/tawnkramer/gym-donkeycar/workflows/CI/badge.svg)](https://github.com/tawnkramer/gym-donkeycar/actions) [![Documentation Status](https://readthedocs.org/projects/gym-donkeycar/badge/?version=latest)](https://gym-donkeycar.readthedocs.io/en/latest/?badge=latest)

Donkey Car OpenAI Gym

  - Free software: MIT license
  - Documentation: <https://gym-donkeycar.readthedocs.io/en/latest/>

## Installation

Download simulator binaries: https://github.com/tawnkramer/gym-donkeycar/releases

Install master version of gym donkey car:

```shell
pip install git+https://github.com/tawnkramer/gym-donkeycar
```

## Example Usage

A short and compact introduction for people who know gym environments,
but want to understand this one. Simple example code:

```python
import os
import gym
import gym_donkeycar
import numpy as np

# SET UP ENVIRONMENT
# You can also launch the simulator separately
# in that case, you don't need to pass a `conf` object
exe_path = f"{PATH_TO_APP}/donkey_sim.exe"
port = 9091

conf = { "exe_path" : exe_path, "port" : port }

env = gym.make("donkey-generated-track-v0", conf=conf)

# PLAY
obs = env.reset()
for t in range(100):
  action = np.array([0.0, 0.5]) # drive straight with small speed
  # execute the action
  obs, reward, done, info = env.step(action)

# Exit the scene
env.close()
```

or if you already launched the simulator:

```python
import gym
import numpy as np

import gym_donkeycar

env = gym.make("donkey-warren-track-v0")

obs = env.reset()
try:
    for _ in range(100):
        # drive straight with small speed
        action = np.array([0.0, 0.5])  
        # execute the action
        obs, reward, done, info = env.step(action)
except KeyboardInterrupt:
    # You can kill the program using ctrl+c
    pass

    # Exit the scene
env.close()
```

- see more examples: https://github.com/tawnkramer/gym-donkeycar/tree/master/examples

## Action space

A permissable action is a numpy array of length two with first steering
and throttle, respectively. E.g. `np.array([0,1])` goes straight at full
speed, `np.array([-5,1])` turns left etc.

Action Space: Box(2,)

Action names: `['steer', 'throttle']`

What you receive back on step:

- obs: The image that the donkey is seeing (np.array shape
  (120,160,3))
- reward: a reward that combines game over, how far from center and
  speed (max=1, min approx -2)
- done: Boolean. Game over if cte > max_cte or hit != "none"
- info contains:
    - cte: Cross track error (how far from center line)
    - positions: x,y,z
    - speed: positive forward, negative backward
    - hit: 'none' if all is good.
    - last_lap_time: time of last successful lap in seconds, 0.0 if there isn't one

Example info:

```python
{'pos': (51.49209, 0.7399381, 117.3004),
 'cte': -5.865292,
 'speed': 9.319956,
 'hit': 'none',
 'last_lap_time': 34.93437361717224}
```

## Environments

- "donkey-warehouse-v0"
- "donkey-generated-roads-v0"
- "donkey-avc-sparkfun-v0"
- "donkey-generated-track-v0"
- "donkey-roboracingleague-track-v0"
- "donkey-waveshare-v0"
- "donkey-minimonaco-track-v0"
- "donkey-warren-track-v0"
- "donkey-thunderhill-track-v0"
- "donkey-circuit-launch-track-v0"


## Codestyle

We are using [black codestyle](https://github.com/psf/black) (max line length of 127 characters) together with [isort](https://github.com/timothycrosley/isort) to sort the imports.

**Please run `make format`** to reformat your code. You can check the codestyle using `make check-codestyle` and `make lint`.

## Tests

Type checking with `pytype`:

```
make type
```

Codestyle check with `black`, `isort` and `flake8`:

```
make check-codestyle
make lint
```

To run `pytype`, `format` and `lint` in one command:
```
make commit-checks
```

Build the documentation:

```
make docs
```


## Credits

Original Source Code

Tawn Kramer - <https://github.com/tawnkramer/gym-donkeycar>

Roma Sokolkov - <https://github.com/r7vme/gym-donkeycar> cloned with
permission from <https://github.com/tawnkramer/sdsandbox>

Maintainer

Maxime Ellerbach - <https://github.com/Maximellerbach/gym-donkeycar>

Release Engineer
Leigh Johnson: https://github.com/leigh-johnson

This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
project template.
