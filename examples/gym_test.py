'''
file: gym_test.py
author: Tawn Kramer
date: 20 October 2018
notes: This will do a basic test of gym_donkeycar environment by
        submitting random input for 3 episodes.
'''
import os
import argparse
import gym
import gym_donkeycar
import time
import random
import uuid

NUM_EPISODES = 3
MAX_TIME_STEPS = 1000


def select_action(env):
    #return env.action_space.sample()
    return [0.0, 0.1] # enable this to test checkpoint failure


def simulate(env):

    for episode in range(NUM_EPISODES):

        # Reset the environment
        obv = env.reset()

        for t in range(MAX_TIME_STEPS):

            # Select an action
            action = select_action(env)

            # execute the action
            obv, reward, done, info = env.step(action)

            if done:
                print("done w episode.", info)
                break


if __name__ == "__main__":

    # Initialize the donkey environment
    # where env_name one of:
    env_list = [
        "donkey-warehouse-v0",
        "donkey-generated-roads-v0",
        "donkey-avc-sparkfun-v0",
        "donkey-generated-track-v0",
        "donkey-mountain-track-v0",
	"donkey-lake-track-v0"
    ]

    parser = argparse.ArgumentParser(description='gym_test')
    parser.add_argument('--sim', type=str, default="sim_path",
                        help='path to unity simulator. maybe be left at default if you would like to start the sim on your own.')
    parser.add_argument('--host', type=str, default="127.0.0.1",
                        help='host to use for tcp')
    parser.add_argument('--port', type=int, default=9091,
                        help='port to use for tcp')
    parser.add_argument('--env_name', type=str, default='donkey-mountain-track-v0',
                        help='name of donkey sim environment', choices=env_list)

    args = parser.parse_args()

    conf = {"exe_path" : args.sim, 
        "host" : args.host,
        "port" : args.port,

        "body_style" : "donkey",
        "body_rgb" : (128, 128, 128),
        "car_name" : "me",
        "font_size" : 100,

        "racer_name" : "test",
        "country" : "USA",
        "bio" : "I am test client",
        "guid" : str(uuid.uuid4()),

        "max_cte" : 20,
        }

    env = gym.make(args.env_name, conf=conf)
    
    simulate(env)

    env.close()

    print("test finished")
