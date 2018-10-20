'''
file: gym_test.py
author: Tawn Kramer
date: 20 October 2018
notes: This will do a basic test of donkey_gym environment by
        submitting random input for 3 episodes.
'''
import os
import argparse
import gym
import donkey_gym
import time
import random

NUM_EPISODES = 3
MAX_TIME_STEPS = 1000

def select_action(env):
    return env.action_space.sample()

def simulate(env):

    for episode in range(NUM_EPISODES):

        # Reset the environment
        obv = env.reset()

        for t in range(MAX_TIME_STEPS):
		
            # Select an action
            action = select_action(env)

            # execute the action
            obv, reward, done, _ = env.step(action)

            if done:
                break

				
				
if __name__ == "__main__":

    # Initialize the donkey environment
    # where env_name one of:    
    env_list = [
       "donkey-warehouse-v0",
       "donkey-generated-roads-v0",
       "donkey-avc-sparkfun-v0",
       "donkey-generated-track-v0"
    ]
    
    parser = argparse.ArgumentParser(description='gym_test')
    parser.add_argument('--sim', type=str, default="sim_path", help='path to unity simulator. maybe be left at default if you would like to start the sim on your own.')
    parser.add_argument('--headless', type=int, default=0, help='1 to supress graphics')
    parser.add_argument('--port', type=int, default=9091, help='port to use for websockets')
    parser.add_argument('--env_name', type=str, default='donkey-generated-track-v0', help='name of donkey sim environment', choices=env_list)
    
    args = parser.parse_args()

    #we pass arguments to the donkey_gym init via these
    os.environ['DONKEY_SIM_PATH'] = args.sim
    os.environ['DONKEY_SIM_PORT'] = str(args.port)
    os.environ['DONKEY_SIM_HEADLESS'] = str(args.headless)

    
    env = gym.make(args.env_name)

    simulate(env)

    env.close()
    
    print("test finished")
