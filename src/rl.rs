use bevy::prelude::{Res, Resource};
use bevy::utils::tracing;
use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::{PyObject, Python};

#[derive(Resource)]
pub struct RLContext {
    pub max_timestep: usize,
    pub elapsed_timestep: usize,
    pub elapsed_episodes: usize,
    pub ctrl_cost: f32,
    pub valid: bool, // used to prevent the RL system from running when the game is paused / resetting
    pub model: PyObject,
    pub initialized: bool,
    // fields updated for training
    pub obs: Option<PyObject>,
    pub action: Option<PyObject>,
    pub log_prob: Option<PyObject>,
    pub critic_val: Option<PyObject>,
}

pub struct ModelArgs {
    pub obs_dim: usize,
    pub act_dim: usize,
    pub ent_coeff: f32,
    pub device: String,
    pub actor_lr: f32,
    pub critic_lr: f32,
    pub timesteps_per_batch: usize,
    pub reward_scale: f32,
}

pub fn is_rl_valid(context: Res<RLContext>) -> bool {
    context.valid
}

pub fn is_rl_invalid(context: Res<RLContext>) -> bool {
    !context.valid
}

impl RLContext {
    pub fn new(max_timestep: usize) -> Self {
        Self {
            max_timestep,
            elapsed_timestep: 0,
            elapsed_episodes: 0,
            ctrl_cost: 0.,
            valid: false,
            model: Python::with_gil(|py| {
                // empty pyobject
                py.None()
            }),
            initialized: false,
            obs: None,
            action: None,
            log_prob: None,
            critic_val: None,
        }
    }

    pub fn init_model(&mut self, args: ModelArgs) -> PyResult<()> {
        if self.initialized {
            return Ok(());
        }
        self.model = Self::instantiate_model(args)?;
        self.initialized = true;
        Ok(())
    }

    // reset the RL context to prepare for a new episode
    pub fn reset(&mut self) {
        self.ctrl_cost = 0.;
        self.valid = false;
        self.obs = None;
        self.action = None;
        self.log_prob = None;
        self.critic_val = None;
    }

    fn instantiate_model(args: ModelArgs) -> PyResult<PyObject> {
        Python::with_gil(|py| -> PyResult<PyObject> {
            let ppo_module = PyModule::from_code(
                py,
                c_str!(include_str!("ppo.py")),
                c_str!("ppo.py"),
                c_str!("ppo"),
            )?;
            let ppo_class = ppo_module.getattr("PPO")?;
            let device_arg = ppo_module
                .getattr("torch")?
                .getattr("device")?
                .call((args.device.into_pyobject(py)?,), None)?;
            let args = PyTuple::new(
                py,
                &[
                    args.obs_dim.into_pyobject(py)?.as_any(),
                    args.act_dim.into_pyobject(py)?.as_any(),
                    args.ent_coeff.into_pyobject(py)?.as_any(),
                    device_arg.as_any(),
                    args.actor_lr.into_pyobject(py)?.as_any(),
                    args.critic_lr.into_pyobject(py)?.as_any(),
                    args.timesteps_per_batch.into_pyobject(py)?.as_any(),
                    args.reward_scale.into_pyobject(py)?.as_any(),
                ],
            )?;
            Ok(ppo_class
                .call(args, None)
                .unwrap()
                .into_pyobject(py)?
                .unbind())
        })
    }

    pub fn learn(&mut self) -> PyResult<()> {
        Python::with_gil(|py| {
            let model = &self.model;
            model.call_method0(py, "learn")?;
            self.elapsed_timestep = 0;
            self.elapsed_episodes += 1;
            tracing::info!("learned");
            Ok(())
        })
    }

    pub fn get_action(&mut self, obs: Vec<f32>) -> PyResult<Vec<f32>> {
        Python::with_gil(|py| {
            let model = &self.model;
            let obs_obj = obs.clone().into_pyobject(py)?;
            let action_and_logprob = model.call_method1(py, "get_action", (obs_obj,))?; // (action, logprob)
            let obs_obj = obs.clone().into_pyobject(py)?.unbind();
            self.obs = Some(obs_obj);
            let action_and_logprob_tuple: &Bound<PyTuple> =
                action_and_logprob.downcast_bound(py)?;
            let action = action_and_logprob_tuple.get_item(0)?.into_pyobject(py)?;
            let logprob = action_and_logprob_tuple.get_item(1)?.into_pyobject(py)?;

            let action: Vec<f32> = action.extract()?;
            let action_obj = action.clone().into_pyobject(py)?.unbind();
            self.action = Some(action_obj);
            self.log_prob = Some(logprob.unbind());

            let obs_obj = obs.clone().into_pyobject(py)?;
            let critic_val = model.call_method1(py, "critic", (obs_obj,))?;
            self.critic_val = Some(critic_val);
            Ok(action)
        })
    }

    pub fn insert_reward(&mut self, reward: f32, done: bool) -> PyResult<()> {
        Python::with_gil(|py| {
            let model = &self.model;

            let obs = self.obs.take().unwrap();
            let action = self.action.take().unwrap();
            let log_prob = self.log_prob.take().unwrap();
            let reward_obj = reward.into_pyobject(py)?;
            let critic_val = self.critic_val.take().unwrap();
            let done = done.into_pyobject(py)?;

            model.call_method1(
                py,
                "insert",
                (
                    obs,
                    action,
                    log_prob,
                    reward_obj,
                    critic_val.call_method0(py, "flatten").unwrap(),
                    done,
                ),
            )?;
            self.elapsed_timestep += 1;
            if self.elapsed_timestep % 1000 == 0 {
                tracing::info!(
                    "time: {}, reward {} inserted",
                    self.elapsed_timestep,
                    reward
                );
            }
            Ok(())
        })
    }
}

impl Default for RLContext {
    fn default() -> Self {
        Self::new(1000)
    }
}
