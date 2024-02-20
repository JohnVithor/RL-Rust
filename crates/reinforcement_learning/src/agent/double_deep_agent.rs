use crate::{
    action_selection::ContinuousObsDiscreteActionSelection, agent::ContinuousObsDiscreteActionAgent,
};
use ndarray::Array1;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::VecDeque;
use tch::{
    nn::{Module, Optimizer, OptimizerConfig, VarStore},
    Device, Tensor,
};

// use super::GetNextQValue;

pub struct Transition {
    pub curr_obs: Tensor,
    pub curr_action: i64,
    pub reward: f32,
    pub done: Tensor,
    pub next_obs: Tensor,
    pub next_action: i64,
}

impl Transition {
    pub fn new(
        curr_obs: &Tensor,
        curr_action: i64,
        reward: f32,
        done: bool,
        next_obs: &Tensor,
        next_action: i64,
    ) -> Self {
        Self {
            curr_obs: curr_obs.shallow_clone(),
            curr_action,
            reward,
            done: Tensor::from(done as i32 as f32),
            next_obs: next_obs.shallow_clone(),
            next_action,
        }
    }
}

pub struct ReplayMemory {
    transitions: VecDeque<Transition>,
    capacity: usize,
    minsize: usize,
    rng: StdRng,
}

impl ReplayMemory {
    pub fn new(capacity: usize, minsize: usize, seed: u64) -> Self {
        Self {
            transitions: VecDeque::new(),
            capacity,
            minsize,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub fn ready(&self) -> bool {
        self.transitions.len() >= self.minsize
    }

    pub fn add(&mut self, transition: Transition) {
        self.transitions.push_back(transition);
        if self.transitions.len() > self.capacity {
            self.transitions.pop_front();
        }
    }

    pub fn sample_batch(
        &mut self,
        size: usize,
    ) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) {
        let index: Vec<usize> = (0..size)
            .map(|_| self.rng.gen_range(0..self.transitions.len()))
            .collect();
        let mut curr_obs: Vec<Tensor> = Vec::new();
        let mut curr_actions: Vec<i64> = Vec::new();
        let mut rewards: Vec<f32> = Vec::new();
        let mut dones: Vec<Tensor> = Vec::new();
        let mut next_obs: Vec<Tensor> = Vec::new();
        let mut next_actions: Vec<i64> = Vec::new();
        index.iter().for_each(|i| {
            let transition = self.transitions.get(*i).unwrap();
            curr_obs.push(transition.curr_obs.shallow_clone());
            curr_actions.push(transition.curr_action);
            rewards.push(transition.reward);
            dones.push(transition.done.shallow_clone());
            next_obs.push(transition.next_obs.shallow_clone());
            next_actions.push(transition.next_action);
        });
        (
            Tensor::stack(&curr_obs, 0),
            Tensor::from_slice(curr_actions.as_slice()).unsqueeze(1),
            Tensor::from_slice(rewards.as_slice()).unsqueeze(1),
            Tensor::stack(&dones, 0).unsqueeze(1),
            Tensor::stack(&next_obs, 0),
            Tensor::from_slice(next_actions.as_slice()).unsqueeze(1),
        )
    }
}

pub struct RunningStat<T> {
    values: VecDeque<T>,
    capacity: usize,
}

impl<T> RunningStat<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            values: VecDeque::new(),
            capacity,
        }
    }

    pub fn add(&mut self, val: T) {
        self.values.push_back(val);
        if self.values.len() > self.capacity {
            self.values.pop_front();
        }
    }

    pub fn average(&self) -> f32
    where
        T: std::iter::Sum,
        T: std::ops::Div<f32, Output = T>,
        T: Clone,
        T: Into<f32>,
        f32: std::iter::Sum<T>,
    {
        let sum: f32 = self.values.iter().cloned().sum();
        sum / (self.capacity as f32)
    }
}

pub struct DoubleDeepAgent {
    pub action_selection: Box<dyn ContinuousObsDiscreteActionSelection>,
    // next_value_function: GetNextQValue,
    pub discount_factor: f32,
    pub optimizer: Optimizer,
    pub policy: Box<dyn Module>,
    pub target_policy: Box<dyn Module>,
    pub policy_vs: VarStore,
    pub target_policy_vs: VarStore,
    pub memory: ReplayMemory,
    pub target_update: usize,
    pub episode_counter: usize,
    pub reward_sum: f32,
    pub stat: RunningStat<f32>,
}

impl DoubleDeepAgent {
    pub fn new(
        action_selection: Box<dyn ContinuousObsDiscreteActionSelection>,
        // next_value_function: GetNextQValue,
        learning_rate: f64,
        discount_factor: f32,
        target_update: usize,
        policy_fn: fn(Device) -> (Box<dyn Module>, VarStore),
        device: Device,
        optimizer: impl OptimizerConfig,
        memory_capacity: usize,
        min_memory_size: usize,
        seed: u64,
    ) -> Self {
        let (policy, policy_vs) = policy_fn(device);
        let (target_policy, mut target_policy_vs) = policy_fn(device);
        target_policy_vs.copy(&policy_vs).unwrap();
        Self {
            action_selection,
            // next_value_function,
            discount_factor,
            episode_counter: 0,
            target_update,
            optimizer: optimizer.build(&policy_vs, learning_rate).unwrap(),
            target_policy,
            policy,
            policy_vs,
            target_policy_vs,
            memory: ReplayMemory::new(memory_capacity, min_memory_size, seed),
            reward_sum: 0.0,
            stat: RunningStat::new(50),
        }
    }
}

impl ContinuousObsDiscreteActionAgent for DoubleDeepAgent {
    fn update(
        &mut self,
        curr_obs: &Array1<f32>,
        curr_action: usize,
        reward: f32,
        terminated: bool,
        next_obs: &Array1<f32>,
        next_action: usize,
    ) -> f32 {
        self.reward_sum += reward;
        let curr_state = Tensor::try_from(curr_obs).unwrap();
        let next_state = Tensor::try_from(next_obs).unwrap();
        let transition = Transition::new(
            &curr_state,
            curr_action as i64,
            reward,
            terminated,
            &next_state,
            next_action as i64,
        );
        self.memory.add(transition);
        if terminated {
            self.episode_counter += 1;
            if self.episode_counter % self.target_update == 0 {
                self.episode_counter = 0;
                self.target_policy_vs.copy(&self.policy_vs).unwrap();
            }
            self.stat.add(self.reward_sum);
            self.reward_sum = 0.0;
            self.action_selection.update(self.stat.average());
            // println!("reward: {}", self.stat.average());
        }
        if self.memory.ready() {
            let (b_curr_obs, b_curr_action, b_reward, b_dones, b_next_obs, _b_next_action) =
                self.memory.sample_batch(128);
            let qvalues = self
                .policy
                .forward(&b_curr_obs)
                .gather(1, &b_curr_action, false);
            let future_q_value: Tensor =
                tch::no_grad(|| self.target_policy.forward(&b_next_obs).max_dim(1, true).0);
            let temporal_difference = b_reward
                + self.discount_factor * (&Tensor::from(1.0) - &b_dones) * (&future_q_value);
            let loss = qvalues.mse_loss(&temporal_difference, tch::Reduction::Mean);
            self.optimizer.zero_grad();
            loss.backward();
            self.optimizer.step();
        }

        // temporal_difference
        0.0
    }

    fn get_action(&mut self, obs: &Array1<f32>) -> usize {
        let values = tch::no_grad(|| self.policy.forward(&Tensor::try_from(obs).unwrap()));
        let values: ndarray::ArrayD<f32> = (&values).try_into().unwrap();
        let values_len: usize = values.len();
        self.action_selection
            .get_action(&values.into_shape(values_len).unwrap())
    }

    fn reset(&mut self) {
        self.action_selection.reset();
    }
}
