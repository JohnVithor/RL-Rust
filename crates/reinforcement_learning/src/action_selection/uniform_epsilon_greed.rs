use environments::env::DiscreteAction;
use rand::{distributions::Uniform, prelude::Distribution};
use std::{fmt::Debug, hash::Hash, rc::Rc};

use crate::utils::argmax;

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

impl<T: Hash + PartialEq + Eq + Clone, A: DiscreteAction> ActionSelection<T, A>
    for UniformEpsilonGreed
{
    fn get_action(&mut self, _obs: &T, values: &[f64; A::RANGE]) -> A {
        if self.should_explore() {
            self.rand_action_selecter
                .sample(&mut rand::thread_rng())
                .into()
        } else {
            argmax(values).into()
        }
    }

    fn update(&mut self) {
        self.decay_epsilon();
    }

    fn get_exploration_probs(&mut self, _obs: &T, values: &[f64; A::RANGE]) -> [f64; A::RANGE] {
        let mut policy_probs: [f64; A::RANGE] = [self.epsilon / values.len() as f64; A::RANGE];
        policy_probs[argmax(values)] = 1.0 - self.epsilon;
        policy_probs
    }

    fn reset(&mut self) {
        self.epsilon = self.initial_epsilon;
    }
}
