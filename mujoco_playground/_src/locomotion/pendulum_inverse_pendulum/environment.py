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
        ctrl_dt=0.02,           # control frequency
        sim_dt=0.002,           # physics frequency (10 substeps)
        episode_length=250,     # 5 seconds per episode
        action_repeat=1,
        action_scale=1.0,
        input_shape=8,          # 6-dimensional observation space
        action_dim=1,             # 1-dimensional action space
        task="swingup",         # "swingup" or "balance"
        reward_config=config_dict.create(
            scales=config_dict.create(
                upright=1.0,        # Tip height reward
                control_cost=-0.01, # Penalize large torques
                velocity_cost=-0.004, # Penalize fast swinging
                continuity_cost=-0.2, # Penalize large changes in torque  (cant bee too high or the action will colapse to 0)
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







class DoublePendulumEnv(mjx_env.MjxEnv):

    def __init__(self, config=None, config_overrides=None):
        if config is None:
            config = default_config()
        super().__init__(config, config_overrides)

        xml_path = (
    mjx_env.ROOT_PATH
    / "locomotion"
    / "pendulum_inverse_pendulum"
    / "xmls"
    / "pendulum_inverse_pendulum.xml"
)
        
        self._xml_path = xml_path.as_posix()

        mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        mj_model.opt.timestep = config.sim_dt

        self._mj_model = mj_model
        self._mjx_model = mjx.put_model(mj_model, impl='jax')
        
        self._step = jax.jit(self._step_impl)

        self._joint_qids  = mjx_env.get_qpos_ids(self._mj_model, ["joint1", "joint2"])
        self._joint_dqids = mjx_env.get_qvel_ids(self._mj_model, ["joint1", "joint2"])
        self._tip_body_id = self._mj_model.body("tip").id
        self._lowers, self._uppers = self._mj_model.actuator_ctrlrange.T

        # Keyframe starting poses
        self._hanging_qpos = jp.array(self._mj_model.keyframe("hanging").qpos)
        self._upright_qpos = jp.array(self._mj_model.keyframe("upright").qpos)
        self.HIST_LEN = 32



    
    def randomize_model(self, rng,
                        mass_randomisation=1.3,
                        damping_randomisation=1.5,
                        friction_randomisation=1.5,
                        armature_randomisation=1,
                        gear_randomisation=1):
        """Returns an mjx.Model with physical params uniformly scaled by
        [1-ratio, 1+ratio] around the values already in the loaded XML."""
        rng_mass, rng_damp, rng_fric, rng_arm, rng_gear = jax.random.split(rng, 5)
        m = self._mjx_model

        def bid(name):
            return mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, name)

        # skip 'pivot' -- fixed to world, scaling its mass has no dynamical effect
        mass_body_ids = jp.array([bid("link1"), bid("link1_tip"), bid("link2"), bid("tip")])
        dof_ids = jp.array(self._joint_dqids)          # [joint1_dof, joint2_dof]
        actuator_ids = jp.arange(m.actuator_gear.shape[0])  # just torque1 here

        def scale(rng_key, values_at_ids, ratio, shape):
            return jax.random.uniform(rng_key, shape, minval=1/ratio, maxval=1*ratio)

        mass_scale = scale(rng_mass, mass_body_ids, mass_randomisation, mass_body_ids.shape)
        new_body_mass = m.body_mass.at[mass_body_ids].multiply(mass_scale)
        new_body_inertia = m.body_inertia.at[mass_body_ids].multiply(mass_scale[:, None])

        damp_scale = scale(rng_damp, dof_ids, damping_randomisation, dof_ids.shape)
        new_damping = m.dof_damping.at[dof_ids].multiply(damp_scale)

        fric_scale = scale(rng_fric, dof_ids, friction_randomisation, dof_ids.shape)
        new_friction = m.dof_frictionloss.at[dof_ids].multiply(fric_scale)

        # armature is 0 in the current XML (never set) -- multiplying 0 by anything stays 0,
        # so this only does something once you give joint1 a nonzero nominal armature
        arm_scale = scale(rng_arm, dof_ids, armature_randomisation, dof_ids.shape)
        new_armature = m.dof_armature.at[dof_ids].multiply(arm_scale)

        gear_scale = scale(rng_gear, actuator_ids, gear_randomisation, actuator_ids.shape)
        new_gear = m.actuator_gear.at[actuator_ids].multiply(gear_scale[:, None])

        return m.replace(
            body_mass=new_body_mass,
            body_inertia=new_body_inertia,
            dof_damping=new_damping,
            dof_frictionloss=new_friction,
            dof_armature=new_armature,
            actuator_gear=new_gear,
        )

    

    def reset(self, rng):
        
        rng, q_rng, v_rng, model_rng = jax.random.split(rng, 4)

        if self._config.task == "balance":
            base_qpos = self._upright_qpos
            noise_scale = 0.2          # Small perturbation near upright
        else:
            base_qpos = self._hanging_qpos
            noise_scale = 3.141/2           # Wider random start for swing-up

        # Create the initial state
        qpos = base_qpos + jax.random.uniform(
            q_rng, base_qpos.shape, minval=-noise_scale, maxval=noise_scale
        )
        qvel = jax.random.uniform(v_rng, (self._mjx_model.nv,), minval=-noise_scale, maxval=noise_scale)

        # Create randomized model for this episode
        model = self.randomize_model(model_rng)

        data = mjx_env.make_data(
            self._mj_model, qpos=qpos, qvel=qvel,
            ctrl=jp.zeros(self._mjx_model.nu),
        )

        info = {"rng": rng, "step": jp.zeros(()), "model": model}
        first_raw = self._raw_obs(data, info)
        info["obs_history"] = jp.tile(first_raw, (self.HIST_LEN, 1))   # (HIST_LEN, obs_dim)
        metrics = {f"reward/{k}": jp.zeros(()) for k in self._config.reward_config.scales}

        obs = self._get_obs(data, info)
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)
    

    def _raw_obs(self, data, info):
        q  = data.qpos[self._joint_qids]   # [θ1, θ2]
        dq = data.qvel[self._joint_dqids]  # [ω1, ω2]

        #add noise
        noise_level = self._config.obs_noise.level
        q_noise_scale = noise_level * self._config.obs_noise.scales.joint_pos
        dq_noise_scale = noise_level * self._config.obs_noise.scales.joint_vel

        key_q, key_dq, new_rng = jax.random.split(info["rng"], 3)
        info["rng"] = new_rng


        q = q + (2 * jax.random.uniform(key_q, q.shape) - 1) * q_noise_scale
        dq = dq + (2 * jax.random.uniform(key_dq, dq.shape) - 1) * dq_noise_scale

        
        obs = jp.asarray([
            jp.cos(q[0]),
            jp.sin(q[0]),           
            jp.cos(q[0]+q[1]),           
            jp.sin(q[0]+q[1]),           
            dq[0]*jp.cos(q[0]),
            dq[0]*jp.sin(q[0]),
            (dq[0] + dq[1])*jp.cos(q[0]+q[1]),
            (dq[0] + dq[1])*jp.sin(q[0]+q[1])

            
            
        ]) 

        
        
        return obs




    def _get_obs(self, data, info):
        """Pushes the new raw obs into the history buffer and returns the
        (possibly lagged/delayed) observation actually fed to the policy."""
        raw = self._raw_obs(data, info)

        # shift buffer left, append newest at the end -- functional, returns new array
        history = jp.concatenate([info["obs_history"][1:], raw[None, :]], axis=0)
        info["obs_history"] = history

        
        lags = jp.array([1])           # indexes of the history to return (0 = most recent, 1 = one step ago, etc.)
        idx = self.HIST_LEN - 1 - lags
        stacked = history[idx]                    
        return stacked.reshape(-1)

    

    def _get_link_positions(self, qpos):
        q = jp.asarray(qpos)
        q1, q2 = float(q[0]), float(q[1])
        pivot = jp.array([0.0, 0.0, 0.0])

        def link_vector(angle):
            return jp.array([-jp.sin(angle), 0.0, -jp.cos(angle)]) * 0.5

        p1 = pivot + link_vector(q1)
        p2 = p1 + link_vector(q1 + q2)
        tip = p2 + link_vector(q1 + q2)
        return pivot, p1, p2, tip

    def simple_render(self, state, ax=None, show=True, figsize=(9, 9)):
        q = state.data.qpos[self._joint_qids]
        pivot, p1, p2, tip = self._get_link_positions(q)

        x = [float(pivot[0]), float(p1[0]), float(p2[0]), float(tip[0])]
        z = [float(pivot[2]), float(p1[2]), float(p2[2]), float(tip[2])]

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            created_fig = True

        ax.plot(x[:2], z[:2], '-o', color='tab:blue', linewidth=3, markersize=8)
        ax.plot(x[1:3], z[1:3], '-o', color='tab:orange', linewidth=3, markersize=8)
        

  
        

        ax.set_title('Double Pendulum Pose')
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_aspect('equal', 'box')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.grid(False)

        if created_fig and show:
            plt.show()


        return ax
    
    def step(self, state, action):
        return self._step(state, action)

    def _step_impl(self, state, action, automatic_reset=False):
        ctrl = jp.clip(action * self._config.action_scale, self._lowers, self._uppers)
        model = state.info["model"]
        data = mjx_env.step(model, state.data, ctrl, self.n_substeps)

        
        upright_reward = ((-2*jp.abs(jp.sin(data.qpos[self._joint_qids[0]]/2))+1)
                           + 2*(-2*jp.abs(jp.sin(data.qpos[self._joint_qids[1]]/2 + data.qpos[self._joint_qids[0]]/2 - jp.pi/2)) + 1))/3

        control_cost   = jp.sum(jp.square(action))
        velocity_cost  = jp.sum(jp.square(data.qvel[self._joint_dqids]))
        continuity_cost = jp.sum(jp.square((ctrl - state.data.ctrl)/(self._uppers - self._lowers)))
        

        reward = (
            upright_reward  * self._config.reward_config.scales.upright +
            control_cost    * self._config.reward_config.scales.control_cost +
            velocity_cost   * self._config.reward_config.scales.velocity_cost +
            continuity_cost * self._config.reward_config.scales.continuity_cost
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
            "reward/continuity_cost": continuity_cost,
        }

        return state.replace(data=data, obs=obs, reward=reward, done=done, info=info, metrics=metrics)
    

    @property
    def xml_path(self):    return self._xml_path
    @property
    def action_size(self): return self._mjx_model.nu      # 2
    @property
    def mj_model(self):    return self._mj_model
    @property
    def mjx_model(self):   return self._mjx_model