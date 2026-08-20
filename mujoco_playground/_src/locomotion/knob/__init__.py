from . import environment as knob_env

_ENVS = {
    "KnobSwingup": knob_env.KnobEnv,
    "KnobBalance": knob_env.KnobEnv,
    "DoublePendulumSwingup": knob_env.DoublePendulumEnv,
    "DoublePendulumBalance": knob_env.DoublePendulumEnv,
}

_CONFIGS = {
    "KnobSwingup": knob_env.default_config,
    "KnobBalance": knob_env.default_config_balance,
    "DoublePendulumSwingup": knob_env.default_config,
    "DoublePendulumBalance": knob_env.default_config_balance,
}