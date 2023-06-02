use rand::distributions::{Distribution, Uniform};
use std::{hash::Hash, cell::RefMut, rc::Rc};
use crate::{env::{ActionSpace}, utils::argmax, policy::Policy, observation::Observation};

use super::ActionSelection;

#[derive(Debug, Clone)]
pub struct UniformEpsilonGreed {
    dist: Uniform<f64>,
    pub initial_epsilon: f64,
    pub epsilon: f64,
    epsilon_decay: f64,
    final_epsilon: f64
}

impl UniformEpsilonGreed {
    pub fn new(epsilon: f64, epsilon_decay: f64, final_epsilon: f64) -> Self {
        return Self {
            dist: Uniform::from(0.0..1.0),
            initial_epsilon: epsilon,
            epsilon,
            epsilon_decay,
            final_epsilon
        }
    }

    pub fn sample(&self) -> bool {
        return self.dist.sample(&mut rand::thread_rng()) < self.epsilon;
    }

    pub fn decay_epsilon(&mut self) {
        let new_epsilon: f64 = self.epsilon - self.epsilon_decay;
        self.epsilon = if self.final_epsilon > new_epsilon {self.epsilon} else {new_epsilon};
    }
}

impl ActionSelection for UniformEpsilonGreed {
    fn get_action(&self, obs: &Rc<dyn Observation>, policy: &mut RefMut<&mut dyn Policy>) -> usize {
        if self.sample() {
            return policy.get_action_space().sample();
        } else {
            match policy.get_ref_if_has(obs) {
                Some(value) => argmax(value),
                None => policy.get_action_space().sample(),
            }
        }
    }

    fn update(&mut self) {
        self.decay_epsilon();
    }

    fn get_exploration_probs(&self, action_space: &ActionSpace) -> Vec<f64> {
        return vec![self.epsilon/action_space.size as f64; action_space.size];
    }

    fn get_exploration_rate(&self) -> f64 {
        return self.epsilon;
    }

    fn reset(&mut self) {
        self.epsilon = self.initial_epsilon;
    }
}