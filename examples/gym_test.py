"""
file: gym_test.py
author: Tawn Kramer
date: 20 October 2018
notes: This will do a basic test of gym_donkeycar environment by
        submitting random input for 3 episodes.
"""

import argparse

import gymnasium as gym

import gym_donkeycar  # noqa: F401

NUM_EPISODES = 3
MAX_TIME_STEPS = 1000


def test_track(env_name, conf, verbose=0, log_interval=10, log_file_handle=None):
    env = gym.make(env_name, conf=conf)

    # make sure you have no track loaded
    exit_scene(env)

    simulate(env, verbose, log_interval, log_file_handle)

    # exit the scene and close the env
    exit_scene(env)
    env.close()


def select_action(env):
    return env.action_space.sample()  # taking random action from the action_space


def simulate(env, verbose=0, log_interval=10, log_file_handle=None):
    """
    Run simulation episodes with configurable logging.

    Args:
        env: Gymnasium environment
        verbose: Verbosity level (0=silent, 1=episode only, 2=interval, 3=every step)
        log_interval: Log every N steps when verbose=2
        log_file_handle: Optional open file handle for logging
    """
    for episode in range(NUM_EPISODES):
        # Reset the environment
        obv, info = env.reset()

        if verbose >= 1:
            print(f"\n{'*'*70}")
            print(f"Starting Episode {episode + 1}/{NUM_EPISODES}")
            print(f"{'*'*70}")

        episode_reward = 0.0

        for step in range(MAX_TIME_STEPS):
            # Select an action
            action = select_action(env)

            # execute the action
            obv, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            # Logging based on verbosity level
            if verbose == 3:  # Every step - detailed
                msg = format_telemetry(step + 1, action, obv, reward, terminated, truncated, info)
                print(msg)
                log_to_file(log_file_handle, msg)
            elif verbose == 2 and (step + 1) % log_interval == 0:  # Every N steps - compact
                msg = format_telemetry_compact(step + 1, action, reward, info)
                print(msg)
                log_to_file(log_file_handle, msg)

            if terminated or truncated:
                if verbose >= 1:
                    summary = format_episode_summary(episode + 1, step + 1, episode_reward, info)
                    print(summary)
                    log_to_file(log_file_handle, summary)
                else:
                    # Original behavior - minimal output
                    print("done w episode.", info)
                break


def exit_scene(env):
    env.unwrapped.viewer.exit_scene()


def format_telemetry(step, action, obv, reward, terminated, truncated, info):
    """
    Format all simulator API data in a readable structure.

    Args:
        step: Current timestep number
        action: [steering, throttle] array
        obv: Observation (image) array
        reward: Float reward value
        terminated: Boolean termination status
        truncated: Boolean truncation status
        info: Dict with all simulator telemetry

    Returns:
        Formatted string with all data
    """
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"STEP {step}")
    lines.append(f"{'='*70}")

    # Action taken
    lines.append(f"\n[ACTION]")
    lines.append(f"  Steering: {action[0]:7.4f}  |  Throttle: {action[1]:7.4f}")

    # Position and orientation
    lines.append(f"\n[POSITION & ORIENTATION]")
    pos = info.get("pos", (0, 0, 0))
    car = info.get("car", (0, 0, 0))
    lines.append(f"  Position: x={pos[0]:7.2f}, y={pos[1]:7.2f}, z={pos[2]:7.2f}")
    lines.append(f"  Orientation: roll={car[0]:7.2f}, pitch={car[1]:7.2f}, yaw={car[2]:7.2f}")

    # Velocity and speed
    lines.append(f"\n[MOTION]")
    vel = info.get("vel", (0, 0, 0))
    lines.append(f"  Speed: {info.get('speed', 0.0):7.2f}")
    lines.append(f"  Forward Vel: {info.get('forward_vel', 0.0):7.2f}")
    lines.append(f"  Velocity: x={vel[0]:7.2f}, y={vel[1]:7.2f}, z={vel[2]:7.2f}")

    # Track position
    lines.append(f"\n[TRACK]")
    lines.append(f"  Cross-Track Error (CTE): {info.get('cte', 0.0):7.4f}")
    lines.append(f"  Hit: {info.get('hit', 'none')}")

    # Sensors
    lines.append(f"\n[SENSORS]")
    gyro = info.get("gyro", (0, 0, 0))
    accel = info.get("accel", (0, 0, 0))
    lines.append(f"  Gyroscope: x={gyro[0]:7.4f}, y={gyro[1]:7.4f}, z={gyro[2]:7.4f}")
    lines.append(f"  Accelerometer: x={accel[0]:7.4f}, y={accel[1]:7.4f}, z={accel[2]:7.4f}")

    # LIDAR
    lidar = info.get("lidar", [])
    if len(lidar) > 0:
        lidar_valid = [d for d in lidar if d >= 0]
        if lidar_valid:
            lines.append(
                f"  LIDAR: {len(lidar_valid)}/{len(lidar)} valid readings, "
                f"min={min(lidar_valid):.2f}, max={max(lidar_valid):.2f}"
            )
        else:
            lines.append(f"  LIDAR: 0/{len(lidar)} valid readings")
    else:
        lines.append(f"  LIDAR: not configured")

    # Lap tracking
    lines.append(f"\n[LAP INFO]")
    lines.append(f"  Lap Count: {info.get('lap_count', 0)}")
    lines.append(f"  Last Lap Time: {info.get('last_lap_time', 0.0):7.2f}s")

    # Observation and reward
    lines.append(f"\n[RL DATA]")
    lines.append(f"  Observation Shape: {obv.shape}")
    lines.append(f"  Reward: {reward:7.4f}")
    lines.append(f"  Terminated: {terminated}  |  Truncated: {truncated}")

    # Optional second camera
    if "image_b" in info:
        lines.append(f"  Secondary Camera: {info['image_b'].shape}")

    lines.append(f"{'='*70}\n")

    return "\n".join(lines)


def format_telemetry_compact(step, action, reward, info):
    """
    Format telemetry in a compact one-line format.

    Args:
        step: Current timestep number
        action: [steering, throttle] array
        reward: Float reward value
        info: Dict with all simulator telemetry

    Returns:
        Compact single-line string
    """
    pos = info.get("pos", (0, 0, 0))
    return (
        f"Step {step:4d} | "
        f"A:[{action[0]:5.2f},{action[1]:5.2f}] | "
        f"Pos:[{pos[0]:5.1f},{pos[1]:5.1f},{pos[2]:5.1f}] | "
        f"Spd:{info.get('speed', 0.0):5.2f} | "
        f"CTE:{info.get('cte', 0.0):6.3f} | "
        f"R:{reward:6.3f} | "
        f"Hit:{info.get('hit', 'none')}"
    )


def format_episode_summary(episode, total_steps, total_reward, info):
    """
    Format episode summary with key statistics.

    Args:
        episode: Episode number
        total_steps: Total steps in episode
        total_reward: Cumulative reward
        info: Final info dict

    Returns:
        Formatted episode summary
    """
    lines = []
    lines.append(f"\n{'#'*70}")
    lines.append(f"EPISODE {episode} COMPLETE")
    lines.append(f"{'#'*70}")
    lines.append(f"  Total Steps: {total_steps}")
    lines.append(f"  Total Reward: {total_reward:.2f}")
    lines.append(f"  Average Reward: {total_reward/total_steps:.4f}")
    lines.append(f"  Termination Reason: {info.get('hit', 'max_cte/other')}")
    lines.append(f"  Laps Completed: {info.get('lap_count', 0)}")
    if info.get("last_lap_time", 0.0) > 0:
        lines.append(f"  Last Lap Time: {info.get('last_lap_time', 0.0):.2f}s")
    lines.append(f"  Final Position: {info.get('pos', (0, 0, 0))}")
    lines.append(f"  Final Speed: {info.get('speed', 0.0):.2f}")
    lines.append(f"{'#'*70}\n")
    return "\n".join(lines)


def log_to_file(file_handle, message):
    """
    Write message to log file.

    Args:
        file_handle: Open file handle
        message: String to write
    """
    if file_handle:
        file_handle.write(message)
        file_handle.write("\n")
        file_handle.flush()


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
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="verbosity level: 0=silent, 1=episode only, 2=every N steps, 3=every step",
    )
    parser.add_argument(
        "--log-interval", type=int, default=10, help="log every N steps when verbose=2 (default: 10)"
    )
    parser.add_argument(
        "--log-file", type=str, default=None, help="optional file path to log data (in addition to screen output)"
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

    # Open log file if specified
    log_file_handle = None
    if args.log_file:
        try:
            log_file_handle = open(args.log_file, "w")
            print(f"Logging to file: {args.log_file}")
        except IOError as e:
            print(f"Warning: Could not open log file {args.log_file}: {e}")
            log_file_handle = None

    if args.env_name == "all":
        for env_name in env_list:
            test_track(env_name, conf, args.verbose, args.log_interval, log_file_handle)

    else:
        test_track(args.env_name, conf, args.verbose, args.log_interval, log_file_handle)

    # Close log file
    if log_file_handle:
        log_file_handle.close()
        print(f"Log written to: {args.log_file}")

    print("test finished")
