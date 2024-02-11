use ndarray::{Array, Array1};
use rand::{distributions::Uniform, prelude::Distribution};
use std::{fmt::Debug, rc::Rc};

use super::ActionSelection;

#[derive(Clone)]
pub struct UniformEpsilonGreed {
    exploration_decider: Uniform<f64>,
    rand_action_selecter: Uniform<usize>,
    pub initial_epsilon: f64,
    pub epsilon: f64,
    epsilon_decay: Rc<dyn Fn(f64) -> f64>,
    final_epsilon: f64,
}

impl Debug for UniformEpsilonGreed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UniformEpsilonGreed")
            .field("exploration_decider", &self.exploration_decider)
            .field("rand_action_selecter", &self.rand_action_selecter)
            .field("initial_epsilon", &self.initial_epsilon)
            .field("epsilon", &self.epsilon)
            .field("final_epsilon", &self.final_epsilon)
            .finish()
    }
}

impl UniformEpsilonGreed {
    pub fn new(
        size: usize,
        epsilon: f64,
        epsilon_decay: Rc<dyn Fn(f64) -> f64>,
        final_epsilon: f64,
    ) -> Self {
        Self {
            exploration_decider: Uniform::from(0.0..1.0),
            rand_action_selecter: Uniform::from(0..size),
            initial_epsilon: epsilon,
            epsilon,
            epsilon_decay,
            final_epsilon,
        }
    }

    fn decay_epsilon(&mut self) {
        let new_epsilon: f64 = (self.epsilon_decay)(self.epsilon);
        self.epsilon = if self.final_epsilon > new_epsilon {
            self.epsilon
        } else {
            new_epsilon
        };
    }

    fn should_explore(&self) -> bool {
        self.epsilon != 0.0
            && self.exploration_decider.sample(&mut rand::thread_rng()) < self.epsilon
    }
}

impl ActionSelection for UniformEpsilonGreed {
    fn get_action(&mut self, _obs: usize, values: &Array1<f64>) -> usize {
        if self.should_explore() {
            self.rand_action_selecter.sample(&mut rand::thread_rng())
        } else {
            values
                .iter()
                .enumerate()
                .skip(1)
                .fold((0, values[0]), |(idx_max, val_max), (idx, val)| {
                    if &val_max > val {
                        (idx_max, val_max)
                    } else {
                        (idx, *val)
                    }
                })
                .0
        }
    }

    fn update(&mut self) {
        self.decay_epsilon();
    }

    fn get_exploration_probs(&mut self, _obs: usize, values: &Array1<f64>) -> Array1<f64> {
        let mut policy_probs: Array1<f64> =
            Array::from_elem((values.len(),), self.epsilon / values.len() as f64);
        let argmax = values
            .iter()
            .enumerate()
            .skip(1)
            .fold((0, values[0]), |(idx_max, val_max), (idx, val)| {
                if &val_max > val {
                    (idx_max, val_max)
                } else {
                    (idx, *val)
                }
            })
            .0;
        policy_probs[argmax] = 1.0 - self.epsilon;
        policy_probs
    }

    fn reset(&mut self) {
        self.epsilon = self.initial_epsilon;
    }
}
