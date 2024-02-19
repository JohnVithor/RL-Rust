use std::collections::VecDeque;

use ndarray::{Array1, Axis};
use rand::{rngs::StdRng, Rng, SeedableRng};
use tch::{
    nn::{Module, Optimizer, OptimizerConfig, VarStore},
    Device, Tensor,
};

use crate::{
    action_selection::ContinuousObsDiscreteActionSelection, agent::ContinuousObsDiscreteActionAgent,
};

use super::GetNextQValue;

pub struct Transition {
    curr_obs: Tensor,
    curr_action: i64,
    reward: f32,
    done: Tensor,
    next_obs: Tensor,
    next_action: i64,
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

pub struct DoubleDeepAgent {
    action_selection: Box<dyn ContinuousObsDiscreteActionSelection>,
    next_value_function: GetNextQValue,
    learning_rate: f64,
    default_values: Array1<f32>,
    default_value: f32,
    discount_factor: f32,
    optimizer: Optimizer,
    policy: Box<dyn Module>,
    target_policy: Box<dyn Module>,
    policy_vs: VarStore,
    target_policy_vs: VarStore,
    memory: ReplayMemory,
}

impl DoubleDeepAgent {
    pub fn new(
        action_selection: Box<dyn ContinuousObsDiscreteActionSelection>,
        next_value_function: GetNextQValue,
        learning_rate: f64,
        default_value: f32,
        discount_factor: f32,
        policy: impl Module + 'static + std::clone::Clone,
        policy_vs: VarStore,
        device: Device,
        optimizer: impl OptimizerConfig,
        memory_capacity: usize,
        min_memory_size: usize,
        seed: u64,
    ) -> Self {
        let mut target_policy_vs = VarStore::new(device);
        target_policy_vs.copy(&policy_vs).unwrap();
        Self {
            action_selection,
            next_value_function,
            learning_rate,
            default_values: Array1::default(0),
            default_value,
            discount_factor,
            optimizer: optimizer.build(&policy_vs, learning_rate).unwrap(),
            target_policy: Box::new(policy.clone()),
            policy: Box::new(policy),
            policy_vs,
            target_policy_vs,
            memory: ReplayMemory::new(memory_capacity, min_memory_size, seed),
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
        if !self.memory.ready() {
            return 0.0;
        }
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
        let (b_curr_obs, b_curr_action, b_reward, b_dones, b_next_obs, b_next_action) =
            self.memory.sample_batch(128);
        let qvalues = self
            .policy
            .forward(&b_curr_obs)
            .gather(1, &b_curr_action, false);
        let next_q_values: Tensor = tch::no_grad(|| self.target_policy.forward(&b_next_obs));
        let next_q_values: ndarray::ArrayD<f32> = (&next_q_values).try_into().unwrap();
        let b_next_action: ndarray::ArrayD<f32> = (&b_next_action).try_into().unwrap();
        let future_q_value = next_q_values.axis_iter(Axis(0)).map(|x| {
            let a = 3;
            3.0
        });
        // next_q_values.apply(m)
        // next_q_values.iter().unwrap().map(|| -> {
        //     (self.next_value_function)()
        // })
        // TODO
        if terminated {
            self.action_selection.update();
        }
        //temporal_difference
        0.0
    }

    fn get_action(&mut self, obs: &Array1<f32>) -> usize {
        let values = tch::no_grad(|| self.policy.forward(&Tensor::try_from(obs).unwrap()));
        let values: ndarray::ArrayD<f32> = (&values).try_into().unwrap();
        self.action_selection
            .get_action(&values.into_shape(1).unwrap())
    }

    fn reset(&mut self) {
        self.action_selection.reset();
    }
}
