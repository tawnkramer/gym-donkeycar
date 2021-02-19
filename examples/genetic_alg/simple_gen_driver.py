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

# In this environment we just want to reward speed, and allow collisions.
# So we just overload the reward and episode over functions accordingly.
def custom_reward(self, done):
    return self.speed

def custom_episode_over(self):
    self.over = False



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
        with self.graph.as_default():
            K.set_session(self.sess)
            self.model._make_predict_function()

        env = gym.make(conf['env_name'], exe_path="remote", host=conf['host'], port=conf['port'])
        
        # setup some custom reward and over functions
        env.set_reward_fn(custom_reward)
        env.set_episode_over_fn(custom_episode_over)

        self.simulate(env)
        env.close()

    def save_model(self, filename):
        with self.graph.as_default():
            K.set_session(self.sess)
            self.model.save(filename)

    def mutate(self):
        with self.graph.as_default():
            K.set_session(self.sess)
            super().mutate()

    def breed(self, agent1, agent2, sess):
        model1, model2 = agent1.model, agent2.model
        jsm = model1.to_json()
        new_model = tf.keras.models.model_from_json(jsm)

        with agent1.graph.as_default():
            K.set_session(agent1.sess)
            w = model1.get_weights()
        
        K.set_session(sess)
        new_model.set_weights(w)

        iLayers = self.conf["layers_to_combine"]
        for iLayer in iLayers:
            layer1 = model1.get_layer(index=iLayer)
            layer2 = model2.get_layer(index=iLayer)
            final_layer = new_model.get_layer(index=iLayer)
            self.merge_layers((final_layer, sess), (layer1, agent1.sess) , (layer2, agent2.sess))

        return new_model

    def merge_layers(self, dest, src1, src2):
        dest_layer, d_sess = dest
        src1_layer, s1_sess = src1
        src2_layer, s2_sess = src2

        K.set_session(s1_sess)
        w1 = src1_layer.get_weights()

        K.set_session(s2_sess)
        w2 = src2_layer.get_weights()

        K.set_session(d_sess)
        res = w1.copy()

        if type(w1) is list:
            half = round(len(w1) / 2)
            res[half:-1] = w2[half:-1]
        else:
            l_indices = np.tril_indices_from(w2)
            res[l_indices] = w2[l_indices]
        dest_layer.set_weights(res)


    def select_action(self, img_arr):
        with self.graph.as_default():
            K.set_session(self.sess)
            one_byte_scale = 1.0 / 255.0
            img_arr = img_arr.astype(np.float32) * one_byte_scale
            img_arr = img_arr.reshape((1,) + img_arr.shape)
            #print("img_arr", type(img_arr), img_arr.shape)
            pred = self.model.predict_on_batch(img_arr)
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

        new_model = self.breed(parent1, parent2, sess)
        agent = KerasDriveAgent(new_model, self.conf, sess)
        agent.mutate()
        return agent



class GeneticPositiveRewardAlg(GeneticAlg):

    def sort_agents(self):
        self.population.sort(key=lambda x: x.get_score(), reverse=True)

    def on_agents_finished(self):
        best_agent = self.population[0]
        best_agent.save_model(self.conf["out_model"])


if __name__ == "__main__":

    # Initialize the donkey environment
    # where env_name one of:
    env_list = [
        "donkey-warehouse-v0",
        "donkey-generated-roads-v0",
        "donkey-avc-sparkfun-v0",
        "donkey-generated-track-v0",
        "donkey-roboracingleague-track-v0",
        "donkey-waveshare-v0",
        "donkey-minimonaco-track-v0",
        "donkey-warren-track-v0"
    ]
	
    parser = argparse.ArgumentParser(description='simple_gen_driver')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='host to use for tcp')
    parser.add_argument('--port', type=int, default=9091, help='port to use for tcp')
    parser.add_argument('--test', action="store_true", help='load the trained model and play')
    parser.add_argument('--num_agents', type=int, default=8, help='how many agents in our population')
    parser.add_argument('--num_iter', type=int, default=8, help='how many generations before we stop')
    parser.add_argument('--max_steps', type=int, default=200, help='how many sim steps before we stop an epoch')
    parser.add_argument('--env_name', type=str, default='donkey-warehouse-v0', help='name of donkey sim environment', choices=env_list)
    parser.add_argument('--in_model', type=str, help='.h5 model produced by donkeycar. expects the default linear model type.')
    parser.add_argument('--out_model', type=str,  default='out.h5', help='.h5 model to save.')

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
        conf["mutation_max"] = 0.1
        conf["mutation_min"] = 0.0
        conf["mutation_decay"] = 1.0
        conf['host'] = args.host
        conf['port'] = args.port
        conf['env_name'] = args.env_name
        conf['MAX_TIME_STEPS'] = args.max_steps
        conf['out_model'] = args.out_model

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
        alg = GeneticPositiveRewardAlg(population=population, conf=conf)
        alg.process(num_iter=args.num_iter)

        ## Our best agent
        agent = alg.population[0]
        print("final score:", agent.score)
