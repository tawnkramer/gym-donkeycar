'''
file: simple_gen_driver.py
author: Tawn Kramer
date: 18 May 2020
notes: a most basic implementation of genetic cross breeding and mutation to attempt to improve
        a neural network. Assumes the standard Keras model from Donkeycar project.
        Lower score means less loss = better.
'''
import os
import random
import time
import argparse
import json
import threading

import gym
import gym_donkeycar
import numpy as np
from PIL import Image

from simple_gen import *

# noisy, noisy tensorflow. we love you.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings  
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf
    from tensorflow.keras import backend as K

tf.logging.set_verbosity(tf.logging.ERROR)


class KerasDriveAgent(KerasNNAgent):
    '''
    Attempt to drive better in sim.
    '''

    def __init__(self, model, conf, sess):
        super().__init__(model, conf)
        self.graph = tf.get_default_graph()
        self.sess = sess

    def begin(self):
        self.score = 0.0
        self.th = threading.Thread(target=self.run)
        self.th.setDaemon(True)
        self.th.start()

    def wait(self):
        self.th.join()

    def run(self):
        conf = self.conf
        K.set_session(self.sess)
        with self.graph.as_default():
            self.model._make_predict_function()

        env = gym.make(conf['env_name'], exe_path="remote", port=conf['port'])
        self.simulate(env)
        env.close()

    def mutate(self):
        K.set_session(self.sess)
        with self.graph.as_default():
            super().mutate()

    def breed(self, agent1, agent2):
        K.set_session(self.sess)
        with self.graph.as_default():
            ret = super().breed(agent1, agent2)
        return ret

    def select_action(self, img_arr):
        K.set_session(self.sess)
        with self.graph.as_default():
            #print("img_arr", type(img_arr), img_arr.shape)
            one_byte_scale = 1.0 / 255.0
            img_arr = img_arr.astype(np.float32) * one_byte_scale
            img_arr = img_arr.reshape((1,) + img_arr.shape)
            pred = self.model.predict(img_arr)
            steering = pred[0][0][0]
            throttle = pred[1][0][0]
            action = [steering, throttle]
            return action

    def simulate(self, env):

        # Reset the environment
        obv = env.reset()

        for _ in range(self.conf['MAX_TIME_STEPS']):

            # Select an action
            action = self.select_action(obv)

            # execute the action
            obv, reward, done, _ = env.step(action)
            
            self.score += reward

            if done:
                break

        print("agent simulate step done. total reward:", self.score)


    def make_new(self, parent1, parent2):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K.set_session(sess)

        new_model = self.breed(parent1, parent2)
        agent = KerasDriveAgent(new_model, self.conf, sess)
        agent.mutate()
        return agent



class GeneticPositiveRewardAlg(GeneticAlg):

    def sort_agents(self):
        self.population.sort(key=lambda x: x.get_score(), reverse=True)




if __name__ == "__main__":

	# Initialize the donkey environment
    # where env_name one of:
    env_list = [
       "donkey-warehouse-v0",
       "donkey-generated-roads-v0",
       "donkey-avc-sparkfun-v0",
       "donkey-generated-track-v0",
       "donkey-mountain-track-v0"
    ]
	
    parser = argparse.ArgumentParser(description='simple_gen_driver')
    parser.add_argument('--port', type=int, default=9091, help='port to use for tcp')
    parser.add_argument('--test', action="store_true", help='load the trained model and play')
    parser.add_argument('--num_agents', type=int, default=4, help='how many agents in our population')
    parser.add_argument('--num_iter', type=int, default=4, help='how many generations before we stop')
    parser.add_argument('--env_name', type=str, default='donkey-mountain-track-v0', help='name of donkey sim environment', choices=env_list)
    parser.add_argument('--in_model', type=str, help='.h5 model produced by donkeycar. expects the default linear model type.')
    parser.add_argument('--out_model', type=str, help='.h5 model to save.')

    args = parser.parse_args()

    env_id = args.env_name

    # only needed if TF==1.13.1
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    

    if args.test:
        pass #todo
        
    else:
    
         # These are the final two dense layers we will mutate. We will use the same two layers we breeding.
        to_mutate = [14, 16]

        conf = { "layers_to_mutate" : to_mutate}
        conf["layers_to_combine"] = to_mutate
        conf["mutation_rate"] = 1.0
        conf["mutation_max"] = 0.3
        conf["mutation_min"] = 0.0
        conf["mutation_decay"] = 1.0
        conf['port'] = args.port
        conf['env_name'] = args.env_name
        conf['MAX_TIME_STEPS'] = 1000
        population = []
        

        for i in range(args.num_agents):
            sess = tf.Session(config=config)
            K.set_session(sess)
            model = tf.keras.models.load_model(os.path.expanduser(args.in_model))
            agent = KerasDriveAgent(model, conf, sess)
            if i > 0:
                agent.mutate()
            population.append(agent)

        ## Try to improve
        alg = GeneticPositiveRewardAlg(population)
        alg.process(num_iter=args.num_iter)

        ## Our best agent
        agent = alg.population[0]
        print("final score:", agent.score)
