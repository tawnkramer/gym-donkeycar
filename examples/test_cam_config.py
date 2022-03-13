"""Test the gym's code for configuring the DonkeyCar's camera settings."""

import argparse
import uuid

import gym
import numpy as np

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
        "donkey-warren-track-v0",
        "donkey-thunderhill-track-v0",
        "donkey-circuit-launch-track-v0",
    ]

    parser = argparse.ArgumentParser(description="gym_test")
    parser.add_argument(
        "--sim",
        type=str,
        default="sim_path",
        help="path to unity simulator. maybe be left at default if you would like to start the sim on your own.",
    )
    parser.add_argument("--port", type=int, default=9091, help="port to use for websockets")
    parser.add_argument(
        "--env_name", type=str, default="donkey-warehouse-v0", help="name of donkey sim environment", choices=env_list
    )

    args = parser.parse_args()

    # SET UP ENVIRONMENT

    cam = (256, 256, 3)

    conf = {
        "exe_path": args.sim,
        "host": "127.0.0.1",
        "port": args.port,
        "body_style": "donkey",
        "body_rgb": (128, 128, 128),
        "car_name": "me",
        "font_size": 100,
        "racer_name": "test",
        "country": "USA",
        "bio": "I am test client",
        "guid": str(uuid.uuid4()),
        "cam_resolution": cam,
        "img_w": cam[0],
        "img_h": cam[1],
        "img_d": cam[2],
    }

    env = gym.make(args.env_name, conf=conf)

    print("Env cam size: {}".format(env.viewer.get_sensor_size()))

    speed = 0.5
    steer = 0.0
    max_steer = 1.0

    # PLAY
    obv = env.reset()
    for t in range(100):
        action = np.array([steer, speed])  # drive straight with small speed
        try:
            obv, reward, done, info = env.step(action)
        except Exception as ex:
            print("Exception: {}".format(ex))

        if obv.shape != cam:
            print("Invalid Image size: {}".format(obv.shape))
        elif t == 10:
            print("Actual camera size: {}".format(obv.shape))

        if done or (info["hit"] is True):
            obv = env.reset()
            print("Exiting d/h: {}/{}".format(done, info["hit"]))
            break

    env.close()
