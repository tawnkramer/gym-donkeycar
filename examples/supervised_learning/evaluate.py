#!/usr/bin/env python
"""
Evaluate
Create a server to accept image inputs and run them against a trained neural network.
This then sends the steering output back to the client.
Author: Tawn Kramer
"""
from __future__ import print_function

import argparse
import base64
import time
from io import BytesIO

import conf
import numpy as np
from donkeycar.utils import linear_unbin
from PIL import Image
from tensorflow.keras.models import load_model

from gym_donkeycar.core.fps import FPSTimer
from gym_donkeycar.core.message import IMesgHandler
from gym_donkeycar.core.sim_client import SimClient


class GifCreator(object):
    def __init__(self, filename):

        self.filename = filename
        self.images = []
        self.every_nth_frame = 4
        self.i_frame = 0

    def add_image(self, image):
        self.i_frame += 1
        if self.i_frame % self.every_nth_frame == 0:
            self.images.append(image)

    def close(self):
        import imageio

        if len(self.images) > 0:
            print("writing movie", self.filename)
            imageio.mimsave(self.filename, self.images)


class DonkeySimMsgHandler(IMesgHandler):
    STEERING = 0
    THROTTLE = 1

    def __init__(self, model, constant_throttle, movie_handler=None):
        self.model = model
        self.constant_throttle = constant_throttle
        self.sock = None
        self.timer = FPSTimer()
        self.image_folder = None
        self.movie_handler = movie_handler
        self.fns = {"telemetry": self.on_telemetry}

    def on_connect(self, client):
        self.client = client
        self.timer.reset()

    def on_recv_message(self, message):
        self.timer.on_frame()
        if "msg_type" not in message:
            print("expected msg_type field")
            print("got:", message)
            return

        msg_type = message["msg_type"]
        if msg_type in self.fns:
            self.fns[msg_type](message)
        else:
            print("unknown message type", msg_type)

    def on_telemetry(self, data):
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)
        self.predict(image_array)

        # maybe write movie
        if self.movie_handler is not None:
            self.movie_handler.add_image(image_array)

    def predict(self, image_array):
        outputs = self.model.predict(image_array[None, :, :, :])
        self.parse_outputs(outputs)

    def parse_outputs(self, outputs):
        res = []
        for iO, output in enumerate(outputs):
            if len(output.shape) == 2:
                if iO == self.STEERING:
                    steering_angle = linear_unbin(output)
                    res.append(steering_angle)
                elif iO == self.THROTTLE:
                    throttle = linear_unbin(output, N=output.shape[1], offset=0.0, R=0.5)
                    res.append(throttle)
                else:
                    res.append(np.argmax(output))
            else:
                for i in range(output.shape[0]):
                    res.append(output[i])

        self.on_parsed_outputs(res)

    def on_parsed_outputs(self, outputs):
        self.outputs = outputs
        steering_angle = 0.0
        throttle = 0.2

        if len(outputs) > 0:
            steering_angle = outputs[self.STEERING]

        if self.constant_throttle != 0.0:
            throttle = self.constant_throttle
        elif len(outputs) > 1:
            throttle = outputs[self.THROTTLE] * conf.throttle_out_scale

        self.send_control(steering_angle, throttle)

    def send_control(self, steer, throttle):
        msg = {"msg_type": "control", "steering": steer.__str__(), "throttle": throttle.__str__(), "brake": "0.0"}
        # print(steer, throttle)
        self.client.queue_message(msg)

    def on_disconnect(self):
        if self.movie_handler:
            self.movie_handler.close()


def go(filename, address, constant_throttle, gif):
    model = load_model(filename, compile=False)

    # In this mode, looks like we have to compile it
    model.compile("sgd", "mse")

    movie_handler = None

    if gif != "none":
        movie_handler = GifCreator(gif)

    # setup the server
    handler = DonkeySimMsgHandler(model, constant_throttle, movie_handler)
    client = SimClient(address, handler)

    while client.is_connected():
        try:
            time.sleep(1.0)
        except KeyboardInterrupt:
            # unless some hits Ctrl+C and then we get this interrupt
            print("stopping")
            break


# ***** main loop *****
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="prediction server")
    parser.add_argument("--model", type=str, help="model filename")
    parser.add_argument("--constant_throttle", type=float, default=0.0, help="apply constant throttle")
    parser.add_argument("--gif", type=str, default="none", help="make animated gif of evaluation")

    args = parser.parse_args()

    address = ("127.0.0.1", 9091)
    go(args.model, address, args.constant_throttle, args.gif)
