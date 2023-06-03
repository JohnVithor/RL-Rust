use rand::distributions::{Distribution, Uniform};
use std::{hash::Hash, cell::RefMut};
use crate::{env::ActionSpace, utils::argmax, policy::Policy};

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

impl<T: Hash+PartialEq+Eq+Clone> ActionSelection<T> for UniformEpsilonGreed {
    fn get_action(&self, obs: &T, policy: &mut RefMut<&mut dyn Policy<T>>) -> usize {
        if self.sample() {
            return policy.get_action_space().sample();
        } else {
            return argmax(policy.predict(obs.clone()));
        }
    }

    fn update(&mut self) {
        self.decay_epsilon();
    }

    fn get_exploration_probs(&self, action_space: &ActionSpace) -> ndarray::Array1<f64> {
        return ndarray::Array1::from_elem(action_space.size, self.epsilon/action_space.size as f64);
    }

    fn get_exploration_rate(&self) -> f64 {
        return self.epsilon;
    }

    fn reset(&mut self) {
        self.epsilon = self.initial_epsilon;
    }
}