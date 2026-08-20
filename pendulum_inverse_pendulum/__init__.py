from . import environment as double_pendulum_env

_ENVS = {
    # ... existing entries ...
    "DoublePendulumSwingup": double_pendulum_env.DoublePendulumEnv,
    "DoublePendulumBalance": double_pendulum_env.DoublePendulumEnv,
}

_CONFIGS = {
    # ... existing entries ...
    "DoublePendulumSwingup": double_pendulum_env.default_config,
    "DoublePendulumBalance": double_pendulum_env.default_config_balance,
}