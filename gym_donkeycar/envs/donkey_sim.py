'''
file: donkey_sim.py
author: Tawn Kramer
date: 2018-08-31
'''

import time
import math
import logging
import base64
from threading import Thread
from io import BytesIO

import numpy as np
from PIL import Image

from gym_donkeycar.core.fps import FPSTimer
from gym_donkeycar.core.message import IMesgHandler
from gym_donkeycar.core.sim_client import SimClient
from gym_donkeycar.envs.donkey_ex import SimFailed

logger = logging.getLogger(__name__)


class DonkeyUnitySimContoller():

    def __init__(self, level, host='127.0.0.1',
                 port=9090, max_cte=5.0, loglevel='INFO', cam_resolution=(120, 160, 3)):

        logger.setLevel(loglevel)

        self.address = (host, port)

        self.handler = DonkeyUnitySimHandler(
            level, max_cte=max_cte,
            cam_resolution=cam_resolution)

        self.client = SimClient(self.address, self.handler)

    def set_car_config(self, body_style, body_rgb, car_name, font_size):
        self.handler.send_car_config(body_style, body_rgb, car_name, font_size)

    def wait_until_loaded(self):
        while not self.handler.loaded:
            logger.warning("waiting for sim to start..")
            time.sleep(3.0)

    def reset(self):
        self.handler.reset()

    def get_sensor_size(self):
        return self.handler.get_sensor_size()

    def take_action(self, action):
        self.handler.take_action(action)

    def observe(self):
        return self.handler.observe()

    def quit(self):
        self.client.stop()

    def render(self, mode):
        pass

    def is_game_over(self):
        return self.handler.is_game_over()

    def calc_reward(self, done):
        return self.handler.calc_reward(done)


class DonkeyUnitySimHandler(IMesgHandler):

    def __init__(self, level, max_cte=5.0, cam_resolution=None):
        self.iSceneToLoad = level
        self.loaded = False
        self.max_cte = max_cte
        self.timer = FPSTimer()

        # sensor size - height, width, depth
        self.camera_img_size = cam_resolution
        self.image_array = np.zeros(self.camera_img_size)
        self.last_obs = None
        self.hit = "none"
        self.cte = 0.0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.speed = 0.0
        self.over = False
        self.fns = {'telemetry': self.on_telemetry,
                    "scene_selection_ready": self.on_scene_selection_ready,
                    "scene_names": self.on_recv_scene_names,
                    "car_loaded": self.on_car_loaded,
                    "aborted": self.on_abort}

    def on_connect(self, client):
        self.client = client

    def on_disconnect(self):
        self.client = None

    def on_abort(self, message):
        self.client.stop()

    def on_recv_message(self, message):
        if 'msg_type' not in message:
            logger.error('expected msg_type field')
            return

        msg_type = message['msg_type']
        if msg_type in self.fns:
            self.fns[msg_type](message)
        else:
            logger.warning(f'unknown message type {msg_type}')

    ## ------- Env interface ---------- ##

    def reset(self):
        logger.debug("reseting")
        self.image_array = np.zeros(self.camera_img_size)
        self.last_obs = self.image_array
        self.hit = "none"
        self.cte = 0.0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.speed = 0.0
        self.over = False
        self.send_reset_car()
        self.timer.reset()
        time.sleep(1)

    def get_sensor_size(self):
        return self.camera_img_size

    def take_action(self, action):
        self.send_control(action[0], action[1])

    def observe(self):
        while self.last_obs is self.image_array:
            time.sleep(1.0 / 120.0)

        self.last_obs = self.image_array
        observation = self.image_array
        done = self.is_game_over()
        reward = self.calc_reward(done)
        info = {'pos': (self.x, self.y, self.z), 'cte': self.cte,
                "speed": self.speed, "hit": self.hit}

        self.timer.on_frame()

        return observation, reward, done, info

    def is_game_over(self):
        return self.over

    ## ------ RL interface ----------- ##

    def calc_reward(self, done):
        if done:
            return -1.0

        if self.cte > self.max_cte:
            return -1.0

        if self.hit != "none":
            return -2.0

        # going fast close to the center of lane yeilds best reward
        return 1.0 - (self.cte / self.max_cte) * self.speed

    ## ------ Socket interface ----------- ##

    def on_telemetry(self, data):

        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))

        # always update the image_array as the observation loop will hang if not changing.
        self.image_array = np.asarray(image)

        self.x = data["pos_x"]
        self.y = data["pos_y"]
        self.z = data["pos_z"]
        self.speed = data["speed"]

        # Cross track error not always present.
        # Will be missing if path is not setup in the given scene.
        # It should be setup in the 4 scenes available now.
        if "cte" in data:
            self.cte = data["cte"]

        # don't update hit once session over
        if self.over:
            return

        self.hit = data["hit"]

        self.determine_episode_over()

    def determine_episode_over(self):
        # we have a few initial frames on start that are sometimes very large CTE when it's behind
        # the path just slightly. We ignore those.
        if math.fabs(self.cte) > 2 * self.max_cte:
            pass
        elif math.fabs(self.cte) > self.max_cte:
            logger.debug(f"game over: cte {self.cte}")
            self.over = True
        elif self.hit != "none":
            logger.debug(f"game over: hit {self.hit}")
            self.over = True

    def on_scene_selection_ready(self, data):
        logger.debug("SceneSelectionReady ")
        self.send_get_scene_names()

    def on_car_loaded(self, data):
        logger.debug("car loaded")
        self.loaded = True

    def on_recv_scene_names(self, data):
        if data:
            names = data['scene_names']
            logger.debug(f"SceneNames: {names}")
            self.send_load_scene(names[self.iSceneToLoad])

    def send_control(self, steer, throttle):
        if not self.loaded:
            return
        msg = {'msg_type': 'control', 'steering': steer.__str__(
        ), 'throttle': throttle.__str__(), 'brake': '0.0'}
        self.queue_message(msg)

    def send_reset_car(self):
        msg = {'msg_type': 'reset_car'}
        self.queue_message(msg)

    def send_get_scene_names(self):
        msg = {'msg_type': 'get_scene_names'}
        self.queue_message(msg)

    def send_load_scene(self, scene_name):
        msg = {'msg_type': 'load_scene', 'scene_name': scene_name}
        self.queue_message(msg)

    def send_car_config(self, body_style, body_rgb, car_name, font_size):
        # body_style = "donkey" | "bare" | "car01" choice of string
        # body_rgb  = (128, 128, 128) tuple of ints
        # car_name = "string less than 64 char"
        msg = {'msg_type': 'car_config',
            'body_style': body_style,
            'body_r' : body_rgb[0].__str__(),
            'body_g' : body_rgb[1].__str__(),
            'body_b' : body_rgb[2].__str__(),
            'car_name': car_name,
            'font_size' : font_size.__str__() }
        self.queue_message(msg)

    def queue_message(self, msg):
        if self.client is None:
            logger.debug(f'skiping: \n {msg}')
            return

        logger.debug(f'sending \n {msg}')
        self.client.queue_message(msg)
