use crate::{
    action_selection::ContinuousObsDiscreteActionSelection,
    agent::ContinuousObsDiscreteActionAgent,
    experience_buffer::{PrioritizedExperienceBuffer, RandomExperienceBuffer},
};
use ndarray::Array1;
use std::collections::VecDeque;
use tch::{
    nn::{Module, Optimizer, OptimizerConfig, VarStore},
    Device, Kind, Tensor,
};
use utils::argmax;

// use super::GetNextQValue;

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
    pub memory: RandomExperienceBuffer,
    pub beta: f64,
    pub target_update: usize,
    pub episode_counter: usize,
    pub reward_sum: f32,
    pub stat: RunningStat<f32>,
    pub device: Device,
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
            memory: RandomExperienceBuffer::new(memory_capacity, min_memory_size, seed, device),
            beta: 0.4,
            reward_sum: 0.0,
            stat: RunningStat::new(50),
            device,
        }
    }

    pub fn memorize(
        &mut self,
        curr_obs: &Array1<f32>,
        curr_action: usize,
        reward: f32,
        terminated: bool,
        next_obs: &Array1<f32>,
        next_action: usize,
    ) {
        let curr_state = Tensor::try_from(curr_obs).unwrap();
        let next_state = Tensor::try_from(next_obs).unwrap();

        self.memory.add(
            &curr_state,
            curr_action as i64,
            reward,
            terminated,
            &next_state,
            next_action as i64,
        );
    }

    pub fn optimize(&mut self, loss: Tensor) {
        self.optimizer.zero_grad();
        loss.backward();
        self.optimizer.step();
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

        let temporal_difference = if self.memory.ready() {
            let (
                b_curr_obs,
                b_curr_action,
                b_reward,
                b_dones,
                b_next_obs,
                _b_next_action,
                // indexes,
                // weights,
            ) = self.memory.sample_batch(128);
            let qvalues = self
                .policy
                .forward(&b_curr_obs.to_device(self.device))
                .gather(1, &b_curr_action.to_device(self.device), false)
                .to_device(self.device);
            let future_q_value: Tensor = tch::no_grad(|| {
                self.target_policy
                    .forward(&b_next_obs.to_device(self.device))
                    .max_dim(1, true)
                    .0
                    .to_device(self.device)
            });
            let temporal_difference = b_reward.to_device(self.device)
                + self.discount_factor
                    * (&Tensor::from(1.0) - &b_dones.to_device(self.device))
                    * (&future_q_value);
            let temporal_difference = temporal_difference
                .to_kind(Kind::Float)
                .to_device(self.device);
            let losses = qvalues.huber_loss(
                &temporal_difference.to_device(self.device),
                tch::Reduction::None,
                1.0,
            );
            let _raw_error = &qvalues - &temporal_difference;
            let loss = (losses).mean(Kind::Float);
            self.optimizer.zero_grad();
            loss.backward();
            self.optimizer.step();
            temporal_difference.double_value(&[0]) as f32
        } else {
            0.0
        };
        if terminated {
            // self.episode_counter += 1;
            // if self.episode_counter % self.target_update == 0 {
            //     self.episode_counter = 0;
            //     self.target_policy_vs.copy(&self.policy_vs).unwrap();
            // }
            // self.stat.add(self.reward_sum);
            // self.reward_sum = 0.0;
            // self.action_selection.update(self.stat.average());
        }
        temporal_difference
    }

    fn get_action(&mut self, obs: &Array1<f32>) -> usize {
        let values = tch::no_grad(|| {
            self.policy
                .forward(&Tensor::try_from(obs).unwrap().to_device(self.device))
        });
        let values: ndarray::ArrayD<f32> = (&values).try_into().unwrap();
        let values_len: usize = values.len();
        self.action_selection
            .get_action(&values.into_shape(values_len).unwrap())
    }

    fn get_best_action(&mut self, obs: &Array1<f32>) -> usize {
        let values = tch::no_grad(|| {
            self.policy
                .forward(&Tensor::try_from(obs).unwrap().to_device(self.device))
        });
        let values: ndarray::ArrayD<f32> = (&values).try_into().unwrap();
        argmax(values.iter())
    }

    fn reset(&mut self) {
        self.action_selection.reset();
    }
}
