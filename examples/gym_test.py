"""
file: gym_test.py
author: Tawn Kramer
date: 20 October 2018
notes: This will do a basic test of gym_donkeycar environment by
        submitting random input for 3 episodes.
"""
import argparse

import gym

import gym_donkeycar  # noqa: F401

NUM_EPISODES = 3
MAX_TIME_STEPS = 1000


def test_track(env_name, conf):
    env = gym.make(env_name, conf=conf)

    # make sure you have no track loaded
    exit_scene(env)

    simulate(env)

    # exit the scene and close the env
    exit_scene(env)
    env.close()


def select_action(env):
    return env.action_space.sample()  # taking random action from the action_space


def simulate(env):

    for _ in range(NUM_EPISODES):

        # Reset the environment
        obv = env.reset()

        for _ in range(MAX_TIME_STEPS):

            # Select an action
            action = select_action(env)

            # execute the action
            obv, reward, done, info = env.step(action)

            if done:
                print("done w episode.", info)
                break


def exit_scene(env):
    env.viewer.exit_scene()


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
    parser.add_argument("--host", type=str, default="127.0.0.1", help="host to use for tcp")
    parser.add_argument("--port", type=int, default=9091, help="port to use for tcp")
    parser.add_argument(
        "--env_name", type=str, default="all", help="name of donkey sim environment", choices=env_list + ["all"]
    )

    args = parser.parse_args()

    conf = {
        "exe_path": args.sim,
        "host": args.host,
        "port": args.port,
        "body_style": "donkey",
        "body_rgb": (128, 128, 128),
        "car_name": "me",
        "font_size": 100,
        "start_delay": 1,
        "max_cte": 5,
        "lidar_config": {
            "deg_per_sweep_inc": 2.0,
            "deg_ang_down": 0.0,
            "deg_ang_delta": -1.0,
            "num_sweeps_levels": 1,
            "max_range": 50.0,
            "noise": 0.4,
            "offset_x": 0.0,
            "offset_y": 0.5,
            "offset_z": 0.5,
            "rot_x": 0.0,
        },
    }

    if args.env_name == "all":
        for env_name in env_list:
            test_track(env_name, conf)

    else:
        test_track(args.env_name, conf)

    print("test finished")
