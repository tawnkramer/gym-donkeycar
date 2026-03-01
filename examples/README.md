# Examples

Sample code to use the gym-donkeycar Gymnasium environment.

**📖 For Gymnasium documentation and API reference, see [gymnasium.farama.org](https://gymnasium.farama.org/)**

## Quick Start

Make sure you have a simulator running, then:

```bash
# Test with random actions
python gym_test.py --sim manual --env_name donkey-generated-track-v0

# Test camera configuration
python test_cam_config.py --sim manual --env_name donkey-warehouse-v0

# Train with PPO (requires: pip install stable-baselines3)
python reinforcement_learning/ppo_train.py --sim manual --env_name donkey-warehouse-v0
```

## Important: Gymnasium Migration

All examples have been updated for Gymnasium API. Key changes:

- Use `import gymnasium as gym` (not `import gym`)
- `step()` returns 5 values: `obs, reward, terminated, truncated, info`
- `reset()` returns 2 values: `obs, info`
- Access custom attributes with `env.unwrapped.viewer`

See the [Gymnasium Migration Guide](https://gymnasium.farama.org/introduction/migration_guide/) for complete migration details.

## Examples Overview

### gym_test.py
Minimal code to test the environment with random actions

### test_cam_config.py
Test custom camera configuration settings

### reinforcement_learning/
Sample code for [reinforcement learning](https://github.com/tawnkramer/gym-donkeycar/tree/master/examples/reinforcement_learning) with gym-donkeycar
- `ppo_train.py`: PPO training with Stable-Baselines3
- `ddqn.py`: Deep Double Q-Learning implementation

### genetic_alg/
Genetic algorithm for evolving neural network controllers
- **Note**: Requires TensorFlow 1.x (legacy, may not work on Python 3.10+)

### supervised_learning/
Sample code for [supervised learning](https://github.com/tawnkramer/gym-donkeycar/tree/master/examples/supervised_learning) from recorded driving data
