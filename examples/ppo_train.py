'''
file: ppo_train.py
author: Tawn Kramer
date: 13 October 2018
notes: ppo2 test from stable-baselines here:
https://github.com/hill-a/stable-baselines
'''
import os
import argparse
import gym
import donkey_gym

from stable_baselines.common.policies import MlpPolicy, CnnPolicy, CnnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines import PPO2

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        env.reset()
        return env
    set_global_seeds(seed)
    return _init


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='ppo_train')
    parser.add_argument('--sim', type=str, default="sim_path", help='path to unity simulator. maybe be left at manual if you would like to start the sim on your own.')
    parser.add_argument('--headless', type=int, default=0, help='1 to supress graphics')
    parser.add_argument('--port', type=int, default=9091, help='port to use for tcp')
    parser.add_argument('--test', action="store_true", help='load the trained model and play')
    parser.add_argument('--multi', action="store_true", help='start multiple sims at once')
    

    args = parser.parse_args()

    if args.sim == "sim_path" and args.multi:
        print("you must supply the sim path with --sim when running multiple environments")
        exit(1)
    
    #we pass arguments to the donkey_gym init via these
    os.environ['DONKEY_SIM_PATH'] = args.sim
    os.environ['DONKEY_SIM_PORT'] = str(args.port)
    os.environ['DONKEY_SIM_HEADLESS'] = str(args.headless)

    env_id = "donkey-generated-track-v0"

    if args.test:

        #Make an environment test our trained policy
        env = gym.make(env_id)
        env = DummyVecEnv([lambda: env])

        model = PPO2.load("ppo_donkey")
    
        obs = env.reset()
        for i in range(1000):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render()

        print("done testing")
        
    else:
    
        if args.multi:

            #setup random offset of network ports
            os.environ['DONKEY_SIM_MULTI'] = '1'

            # Number of processes to use
            num_cpu = 4

            # Create the vectorized environment
            env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

            #create recurrent policy
            model = PPO2(CnnLstmPolicy, env, verbose=1)

        else:

            #make gym env
            env = gym.make(env_id)

            # Create the vectorized environment
            env = DummyVecEnv([lambda: env])

            #create cnn policy
            model = PPO2(CnnPolicy, env, verbose=1)


        #set up model in learning mode with goal number of timesteps to complete
        model.learn(total_timesteps=10000)

        obs = env.reset()
        
        for i in range(1000):
            
            action, _states = model.predict(obs)
            
            obs, rewards, dones, info = env.step(action)
            
            try:
                env.render()
            except Exception as e:
                print(e)
                print("failure in render, continuing...")
                
            if i % 100 == 0:
                print('saving...')
                model.save("ppo_donkey")

        # Save the agent
        model.save("ppo_donkey")
        print("done training")


    env.close()
