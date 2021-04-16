"""
file: simple_gen.py
author: Tawn Kramer
date: 17 May 2020
notes: a most basic implementation of genetic cross breeding and mutation to attempt to improve
        a neural network. Assumes the standard Keras model from Donkeycar project.
        Lower score means less loss = better.
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
from PIL import Image

# noisy, noisy tensorflow. we love you.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)


class IAgent:
    def begin(self):
        pass

    def wait(self):
        pass

    def get_score(self):
        pass

    def make_new(self, parent1, parent2):
        return IAgent()


class GeneticAlg:
    def __init__(self, population, conf=None):
        self.population = population
        if conf is None:
            conf = {}
        self.conf = conf

    def finished(self):
        return False

    def process(self, num_iter):
        iIter = 0
        while not self.finished() and iIter < num_iter:
            print("starting epoch", iIter)
            s = time.time()
            self.evaluate_agents()
            self.on_agents_finished()

            e = time.time() - s
            self.breed_agents()
            iIter += 1
            d = time.time() - s
            # Time per iteration getting worse?!
            print("finish epoch", iIter)
            print("Iter %d eval time: %f total time: %f" % (iIter, e, d))

    def on_agents_finished(self):
        pass

    def evaluate_agents(self):
        for agent in self.population:
            agent.begin()

        for agent in self.population:
            agent.wait()

        self.sort_agents()

        # progress
        print("scores:", [a.score for a in self.population])

    def how_many_to_keep(self):
        return round(len(self.population) / 4) + 1

    def breed_agents(self):
        """
        keep the best N of our population and replace the rest
        with versions cross bred from other agents.
        """

        keep = self.how_many_to_keep()
        num_new = len(self.population) - keep
        pop_to_keep = self.population[0:keep]
        new_population = []

        for _ in range(num_new):
            p1, p2 = self.select_parents()
            new_agent = p1.make_new(p1, p2)
            new_agent.mutate()
            new_population.append(new_agent)

        self.population = pop_to_keep + new_population

    def sort_agents(self):
        self.population.sort(key=lambda x: x.get_score(), reverse=False)

    def select_pop_index(self):
        r = np.random.uniform(low=0.0, high=1.0)
        N = len(self.population)
        iP = round(r * N) % N
        return iP

    def select_parents(self):
        iP1 = self.select_pop_index()
        iP2 = self.select_pop_index()

        # hack, always select the best 2
        # iP1 = 0
        # iP2 = 1

        # lets make sure parents are not the same
        while iP2 == iP1:
            iP2 = self.select_pop_index()

        return self.population[iP1], self.population[iP2]


class NNAgent(IAgent):
    def __init__(self, model, conf):
        self.model = model
        self.score = 0.0
        self.conf = conf

    def begin(self):
        self.score = 0.0

    def wait(self):
        pass

    def get_score(self):
        return self.score

    def mutate(self):
        pass

    def breed(self, agent1, agent2):
        return agent1.model

    def make_new(self, parent1, parent2):
        new_model = self.breed(parent1, parent2)
        agent = NNAgent(new_model, self.conf)
        agent.mutate()
        return agent


class KerasNNAgent(NNAgent):
    def __init__(self, model, conf):
        super().__init__(model, conf)
        self.mutation_rate = conf["mutation_rate"]

    def mutate(self):
        layers_to_mutate = self.conf["layers_to_mutate"]

        for iLayer in layers_to_mutate:
            layer = self.model.get_layer(index=iLayer)
            w = layer.get_weights()
            self.modify_weights(w)
            layer.set_weights(w)

        self.decay_mutations()

    def rand_float(self, mn, mx):
        return float(np.random.uniform(mn, mx, 1)[0])

    def modify_weights(self, w):
        mx = self.conf["mutation_max"]
        mn = self.conf["mutation_min"]
        mag = self.rand_float(mn, mx)

        for iArr, arr in enumerate(w):
            val = self.rand_float(0.0, 1.0)
            if val > self.mutation_rate:
                continue

            random_values = np.random.uniform(-mag, mag, arr.shape)
            arr = arr + random_values
            w[iArr] = arr
        return w

    def decay_mutations(self):
        self.conf["mutation_max"] *= self.conf["mutation_decay"]

    def breed(self, agent1, agent2):
        model1, model2 = agent1.model, agent2.model
        jsm = model1.to_json()
        new_model = tf.keras.models.model_from_json(jsm)
        new_model.set_weights(model1.get_weights())

        iLayers = self.conf["layers_to_combine"]
        for iLayer in iLayers:
            layer1 = model1.get_layer(index=iLayer)
            layer2 = model2.get_layer(index=iLayer)
            final_layer = new_model.get_layer(index=iLayer)
            self.merge_layers(final_layer, layer1, layer2)

        return new_model

    def merge_layers(self, dest_layer, src1_layer, src2_layer):
        w1 = src1_layer.get_weights()
        w2 = src2_layer.get_weights()
        res = w1.copy()
        if type(w1) is list:
            half = round(len(w1) / 2)
            res[half:-1] = w2[half:-1]
        else:
            l_indices = np.tril_indices_from(w2)
            res[l_indices] = w2[l_indices]
        dest_layer.set_weights(res)


class KerasNNImageAgent(KerasNNAgent):
    """
    Given an image and a target prediction, make an agent that will
    optimize for score of target.
    """

    def __init__(self, model, conf):
        super().__init__(model, conf)
        self.image = conf["image"]
        self.target = conf["target"]

    def begin(self):
        pred = self.model.predict(self.image)
        self.score = np.sum(np.absolute(pred - self.target))

    def make_new(self, parent1, parent2):
        new_model = self.breed(parent1, parent2)
        agent = KerasNNImageAgent(new_model, self.conf)
        agent.mutate()
        return agent


def test_image_agent(model_filename, record_filename, num_agents, num_iter):
    with open(os.path.expanduser(record_filename), "r") as fp:
        record = json.load(fp)
    img_filename = os.path.join(os.path.dirname(record_filename), record["cam/image_array"])
    img = Image.open(os.path.expanduser(img_filename))
    img_arr = np.array(img)

    # Our model was trained with this normalization scale on data.
    one_byte_scale = 1.0 / 255.0
    img_arr = img_arr.astype(np.float32) * one_byte_scale
    img_arr = img_arr.reshape((1,) + img_arr.shape)
    steering = record["user/angle"]
    throttle = record["user/throttle"]
    target = np.array([np.array([[steering]]), np.array([[throttle]])])

    # These are the final two dense layers we will mutate. We will use the same two layers we breeding.
    to_mutate = [14, 16]

    conf = {"layers_to_mutate": to_mutate}
    conf["layers_to_combine"] = to_mutate
    conf["mutation_rate"] = 1.0
    conf["mutation_max"] = 0.3
    conf["mutation_min"] = 0.0
    conf["mutation_decay"] = 1.0
    conf["image"] = img_arr
    conf["target"] = target
    population = []

    for i in range(num_agents):
        model = tf.keras.models.load_model(os.path.expanduser(model_filename))
        agent = KerasNNImageAgent(model, conf)
        if i > 0:
            agent.mutate()
        population.append(agent)

    # Some initial state
    print("target: steering: %f throttle: %f" % (target[0][0][0], target[1][0][0]))
    agent = population[0]
    agent.begin()
    print("initial score:", agent.score)
    pred = agent.model.predict(img_arr)
    print("initial pred", pred[0][0], pred[1][0])

    # Try to improve
    alg = GeneticAlg(population)
    alg.process(num_iter=num_iter)

    # Our best agent
    agent = alg.population[0]
    print("final score:", agent.score)
    pred = agent.model.predict(img_arr)
    print("final pred", pred[0][0], pred[1][0])


if __name__ == "__main__":
    # Example: python ~\projects\gym-donkeycar\examples\genetic_alg\simple_gen.py
    # --model models\lane_keeper.h5 --record data\tub_6_20-05-16\record_2000.json
    parser = argparse.ArgumentParser(description="simple_gen")
    parser.add_argument("--model", type=str, help=".h5 model produced by donkeycar. expects the default linear model type.")
    parser.add_argument("--record", type=str, help="donkey json record to use for training")
    parser.add_argument("--num_agents", type=int, default=8, help="how many agents in our population")
    parser.add_argument("--num_iter", type=int, default=8, help="how many generations before we stop")

    args = parser.parse_args()

    # only needed if TF==1.13.1
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    # K.set_session(sess)

    test_image_agent(
        model_filename=args.model, record_filename=args.record, num_agents=args.num_agents, num_iter=args.num_iter
    )
