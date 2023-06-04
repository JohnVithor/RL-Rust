use rand::{distributions::Uniform, prelude::Distribution};
use std::hash::Hash;

use crate::utils::argmax;

use super::ActionSelection;

#[derive(Debug, Clone)]
pub struct UniformEpsilonGreed<const COUNT: usize> {
    exploration_decider: Uniform<f64>,
    rand_action_selecter: Uniform<usize>,
    pub initial_epsilon: f64,
    pub epsilon: f64,
    epsilon_decay: f64,
    final_epsilon: f64,
}

impl<const COUNT: usize> UniformEpsilonGreed<COUNT> {
    pub fn new(epsilon: f64, epsilon_decay: f64, final_epsilon: f64) -> Self {
        return Self {
            exploration_decider: Uniform::from(0.0..1.0),
            rand_action_selecter: Uniform::from(0..COUNT),
            initial_epsilon: epsilon,
            epsilon,
            epsilon_decay,
            final_epsilon,
        };
    }

    fn decay_epsilon(&mut self) {
        let new_epsilon: f64 = self.epsilon - self.epsilon_decay;
        self.epsilon = if self.final_epsilon > new_epsilon {
            self.epsilon
        } else {
            new_epsilon
        };
    }

    fn should_explore(&self) -> bool {
        return self.epsilon != 0.0
            && self.exploration_decider.sample(&mut rand::thread_rng()) < self.epsilon;
    }
}

impl<T: Hash + PartialEq + Eq + Clone, const COUNT: usize> ActionSelection<T, COUNT>
    for UniformEpsilonGreed<COUNT>
{
    fn get_action(&mut self, _obs: &T, values: &[f64; COUNT]) -> usize {
        if self.should_explore() {
            return self.rand_action_selecter.sample(&mut rand::thread_rng());
        } else {
            return argmax(values);
        }
    }

    fn update(&mut self) {
        self.decay_epsilon();
    }

    fn get_exploration_probs(&mut self, _obs: &T, values: &[f64; COUNT]) -> [f64; COUNT] {
        let mut policy_probs: [f64; COUNT] = [self.epsilon / COUNT as f64; COUNT];
        policy_probs[argmax(values)] = 1.0 - self.epsilon;
        return policy_probs;
    }

    fn reset(&mut self) {
        self.epsilon = self.initial_epsilon;
    }
}
