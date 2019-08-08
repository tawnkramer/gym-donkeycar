# donkey_gym

OpenAI gym environment for donkeycar simulator.

# Legal notice

Package is under MIT license. *Authored by Tawn Kramer and original sources located [here](https://github.com/tawnkramer/sdsandbox/tree/donkey/src/donkey_gym)*. 


## Donkeycar Simulator

The [donkeycar simulator](https://github.com/tawnkramer/sdsandbox/tree/donkey) is a [Unity](http://www.unity3d.com) application using [Nvidia's PhysX](https://developer.nvidia.com/physx-sdk) to simulate a [donkeycar](http://www.donkeycar.com) driving in various environments. The simulator connects to an external process via tcp socket. It sends telemetry to and recieves control from the external process. 


## Installation

* git clone https://github.com/tawnkramer/donkey_gym
* pip install -e donkey_gym

## Simulator Binaries

* Download [simulator binaries](https://github.com/tawnkramer/donkey_gym/releases).

## Examples

* see [examples](https://github.com/tawnkramer/donkey_gym/tree/master/examples) of use.


## Environment quickstart:
A short and compact introduction for people who know gym environments, but want to understand this one. 
Simple example code:
```
import os
import gym
import donkey_gym
import numpy as np

#%% SET UP ENVIRONMENT
os.environ['DONKEY_SIM_PATH'] = f"{PATH_TO_APP}/donkey_sim.app/Contents/MacOS/donkey_sim"
os.environ['DONKEY_SIM_PORT'] = str(9000)
os.environ['DONKEY_SIM_HEADLESS'] = str(0) # "1" is headless

env = gym.make("donkey-warehouse-v0")

#%% PLAY
obv = env.reset()
for t in range(100):
    action = np.array([0.0,0.5]) # drive straight with small speed
    # execute the action
    obv, reward, done, info = env.step(action)
```

### Action space
A permissable action is a numpy array of length two with first steering and throttle, respectively. E.g. np.array([0,1]) goes straight at full speed, np.array([-5,1]) turns left etc.

Action Space: Box(2,)  
Action names: ['steer', 'throttle']


### What you receive back on step:
- obv: The image that the donkey is seeing (np.array shape (120,160,3))
- reward: a reward that combines game over, how far from center and speed (max=1, min approx -2)
- done: Boolean. Game over if cte > max_cte or hit != "none"

- info contains:
  - cte: Cross track error (how far from center line)
  - positions: x,y,z
  - speed: positive forward, negative backward
  - hit: 'none' if all is good.

example info:
```
{'pos': (51.49209, 0.7399381, 117.3004),
 'cte': -5.865292,
 'speed': 9.319956,
 'hit': 'none'}
```
