import random

from ml_collections import config_dict
from mujoco_playground._src import mjx_env
import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
import mujoco
from mujoco import mjx




def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,           # 50 Hz control frequency
        sim_dt=0.002,           # 500 Hz physics (10 substeps)
        episode_length=500,     # 10 seconds per episode
        action_repeat=1,
        action_scale=1.0,
        input_shape=jp.array([4], dtype=jp.int32),           # 1 joint angle + rate + sin/cos encoding
        action_dim=jp.array([1], dtype=jp.int32),             # 1-dimensional action space
        task="setpoint1",         # 1 2 3
        reward_config=config_dict.create(
            scales=config_dict.create(
                upright=1.0,        # Angle-stability reward
                control_cost=-0.01, # Penalize large torques
                velocity_cost=-0.001, # Penalize fast swinging
            ),
        ),
        obs_noise=config_dict.create(
            level=1.0,
            scales=config_dict.create(
                joint_pos=0.02,
                joint_vel=0.05,
            ),
        ),
        naconmax=50,
        njmax=50,
    )

def default_config_balance() -> config_dict.ConfigDict:
    config = default_config()
    config.task = "balance"
    return config

class KnobEnv(mjx_env.MjxEnv):

    def __init__(self, config=None, config_overrides=None):
        if config is None:
            config = default_config()
        super().__init__(config, config_overrides)

        xml_path = (
            mjx_env.ROOT_PATH
            / "locomotion"
            / "knob"
            / "xmls"
            / "knob.xml"
        )

        self._xml_path = xml_path.as_posix()

        mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        mj_model.opt.timestep = config.sim_dt

        self._mj_model = mj_model
        self._mjx_model = mjx.put_model(mj_model, impl='jax')

        self._step = jax.jit(self._step_impl)

        self._joint_qids = mjx_env.get_qpos_ids(self._mj_model, ["joint1"])
        self._joint_dqids = mjx_env.get_qvel_ids(self._mj_model, ["joint1"])
        self._tip_body_id = self._mj_model.body("tip").id
        self._lowers, self._uppers = self._mj_model.actuator_ctrlrange.T

        
        self._setpoint1_qpos = jp.array(self._mj_model.keyframe("setpoint1").qpos)
        self._setpoint2_qpos = jp.array(self._mj_model.keyframe("setpoint2").qpos)
        self._setpoint3_qpos = jp.array(self._mj_model.keyframe("setpoint3").qpos)
        self._hanging_qpos = jp.array(self._mj_model.keyframe("hanging").qpos)



    

    def reset(self, rng=jax.random.PRNGKey(random.randint(0, 100000))):
        rng, q_rng, v_rng = jax.random.split(rng, 3)

        if self._config.task == "setpoint1":
            base_qpos = self._setpoint1_qpos
            noise_scale = 0.0
        elif self._config.task == "setpoint2":
            base_qpos = self._setpoint2_qpos
            noise_scale = 0.0
        elif self._config.task == "setpoint3":
            base_qpos = self._setpoint3_qpos
            noise_scale = 0.0
        elif self._config.task == "hanging":
            base_qpos = self._hanging_qpos
            noise_scale = 0.0

        qpos = base_qpos + jax.random.uniform(
            q_rng, base_qpos.shape, minval=-noise_scale, maxval=noise_scale
        )
        qvel = jax.random.uniform(
            v_rng,
            (self._mjx_model.nv,),
            minval=-noise_scale,
            maxval=noise_scale,
        )

        data = mjx_env.make_data(
            self._mj_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=jp.zeros(self._mjx_model.nu),
        )

        info = {"rng": rng, "step": jp.zeros(())}
        metrics = {f"reward/{k}": jp.zeros(()) for k in self._config.reward_config.scales}

        obs = self._get_obs(data, info)
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)
    

    def _get_obs(self, data, info):
        del info
        q = data.qpos[self._joint_qids]
        dq = data.qvel[self._joint_dqids]

        obs = jp.concatenate([
            q,
            dq,
            jp.cos(q),
            jp.sin(q),
        ])

        return obs

    def _get_link_positions(self, qpos):
        q = jp.asarray(qpos)
        angle = float(q[0])
        pivot = jp.array([0.0, 0.0, 0.0])

        def link_vector(angle):
            return jp.array([-jp.sin(angle), 0.0, -jp.cos(angle)]) * 0.5

        tip = pivot + link_vector(angle)
        return pivot, tip

    def simple_render(self, state, ax=None, show=True, figsize=(9, 9)):
        q = state.data.qpos[self._joint_qids]
        pivot, tip = self._get_link_positions(q)

        x = [float(pivot[0]), float(tip[0])]
        z = [float(pivot[2]), float(tip[2])]

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            created_fig = True

        ax.plot(x, z, '-o', color='tab:blue', linewidth=3, markersize=8)
        ax.scatter([float(pivot[0])], [float(pivot[2])], color='black', s=35)
        ax.scatter([float(tip[0])], [float(tip[2])], color='tab:red', s=45)

        ax.set_title('Knob Pose')
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_aspect('equal', 'box')
        ax.set_xlim(-0.75, 0.75)
        ax.set_ylim(-0.75, 0.75)
        ax.grid(False)

        if created_fig and show:
            plt.show()

        return ax
    
    def step(self, state, action):
        return self._step(state, action)

    def _step_impl(self, state, action):
        ctrl = jp.clip(action * self._config.action_scale, self._lowers, self._uppers)
        data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)

        angle = data.qpos[self._joint_qids[0]]
        upright_reward = jp.cos(angle)
        control_cost = jp.sum(jp.square(action))
        velocity_cost = jp.sum(jp.square(data.qvel[self._joint_dqids]))

        reward = (
            upright_reward * self._config.reward_config.scales.upright +
            control_cost * self._config.reward_config.scales.control_cost +
            velocity_cost * self._config.reward_config.scales.velocity_cost
        )

        done = jp.logical_or(
            jp.any(jp.isnan(data.qpos)),
            state.info["step"] + 1 >= self._config.episode_length,
        ).astype(jp.float32)

        info = {**state.info, "step": state.info["step"] + 1}
        obs = self._get_obs(data, info)

        metrics = {
            **state.metrics,
            "reward/upright": upright_reward,
            "reward/control_cost": control_cost,
            "reward/velocity_cost": velocity_cost,
        }

        return state.replace(data=data, obs=obs, reward=reward, done=done, info=info, metrics=metrics)
    

    @property
    def xml_path(self):
        return self._xml_path

    @property
    def action_size(self):
        return self._mjx_model.nu

    @property
    def mj_model(self):
        return self._mj_model

    @property
    def mjx_model(self):
        return self._mjx_model


DoublePendulumEnv = KnobEnv