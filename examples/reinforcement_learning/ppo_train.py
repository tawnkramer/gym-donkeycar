"""
file: ppo_train.py
author: Tawn Kramer
date: 13 October 2018
notes: ppo2 test from stable-baselines here:
https://github.com/hill-a/stable-baselines
"""

import argparse
import uuid

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

import gym_donkeycar  # noqa: F401

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
        "donkey-mountain-track-v0",
    ]

    parser = argparse.ArgumentParser(description="ppo_train")
    parser.add_argument(
        "--sim",
        type=str,
        default="sim_path",
        help="path to unity simulator. maybe be left at manual if you would like to start the sim on your own.",
    )
    parser.add_argument("--port", type=int, default=9091, help="port to use for tcp")
    parser.add_argument("--test", action="store_true", help="load the trained model and play")
    parser.add_argument("--multi", action="store_true", help="start multiple sims at once")
    parser.add_argument(
        "--env_name", type=str, default="donkey-warehouse-v0", help="name of donkey sim environment", choices=env_list
    )
    parser.add_argument(
        "--timesteps", type=int, default=10000, help="number of timesteps to train for (default: 10000)"
    )
    parser.add_argument(
        "--load", type=str, default=None, help="path to pretrained model to load and continue training"
    )

    args = parser.parse_args()

    if args.sim == "sim_path" and args.multi:
        print("you must supply the sim path with --sim when running multiple environments")
        exit(1)

    env_id = args.env_name

    conf = {
        "exe_path": args.sim,
        "host": "127.0.0.1",
        "port": args.port,
        "body_style": "donkey",
        "body_rgb": (128, 128, 128),
        "car_name": "me",
        "font_size": 100,
        "racer_name": "PPO",
        "country": "USA",
        "bio": "Learning to drive w PPO RL",
        "guid": str(uuid.uuid4()),
        "max_cte": 10,
    }

    if args.test:
        # Make an environment test our trained policy
        env = gym.make(args.env_name, conf=conf)

        model = PPO.load("ppo_donkey")

        obs, info = env.reset()
        for _ in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            if terminated or truncated:
                obs, info = env.reset()

        print("done testing")

    else:
        # make gym env
        env = gym.make(args.env_name, conf=conf)

        # create or load model
        if args.load:
            print(f"Loading pretrained model from {args.load}")
            model = PPO.load(args.load, env=env)
            print("Continuing training from pretrained model...")
        else:
            print("Creating new model from scratch...")
            model = PPO("CnnPolicy", env, verbose=1)

        # set up checkpoint callback to save model every 5000 steps
        checkpoint_callback = CheckpointCallback(
            save_freq=5000,
            save_path="./checkpoints/",
            name_prefix="ppo_donkey",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )

        # set up model in learning mode with goal number of timesteps to complete
        print(f"Training for {args.timesteps} timesteps...")
        print("Model will be saved every 5000 steps to ./checkpoints/")
        model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback)

        obs, info = env.reset()

        for i in range(1000):
            action, _states = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)

            try:
                env.render()
            except Exception as e:
                print(e)
                print("failure in render, continuing...")

            if terminated or truncated:
                obs, info = env.reset()

        # Save the agent
        model.save("ppo_donkey")
        print("done training, model saved as ppo_donkey.zip")

    env.close()
