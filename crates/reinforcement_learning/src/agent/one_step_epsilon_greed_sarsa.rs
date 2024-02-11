use std::collections::HashMap;

use ndarray::{Array, Array1};
use rand::distributions::{Distribution, Uniform};

use crate::agent::DiscreteAgent;

pub struct OneStepEpsilonGreedSarsa {
    exploration_decider: Uniform<f64>,
    rand_action_selecter: Uniform<usize>,
    epsilon: f64,
    initial_epsilon: f64,
    epsilon_decay: f64,
    final_epsilon: f64,
    learning_rate: f64,
    default_value: Array1<f64>,
    discount_factor: f64,
    policy: HashMap<usize, Array1<f64>>,
}

impl OneStepEpsilonGreedSarsa {
    pub fn new(
        n_actions: usize,
        epsilon: f64,
        epsilon_decay: f64,
        final_epsilon: f64,
        learning_rate: f64,
        default_value: f64,
        discount_factor: f64,
    ) -> Self {
        Self {
            exploration_decider: Uniform::from(0.0..1.0),
            rand_action_selecter: Uniform::from(0..n_actions),
            epsilon,
            initial_epsilon: epsilon,
            epsilon_decay,
            final_epsilon,
            learning_rate,
            default_value: Array::from_elem((n_actions,), default_value),
            discount_factor,
            policy: HashMap::new(),
        }
    }
}
impl DiscreteAgent for OneStepEpsilonGreedSarsa {
    fn update(
        &mut self,
        curr_obs: usize,
        curr_action: usize,
        reward: f64,
        terminated: bool,
        next_obs: usize,
        next_action: usize,
    ) -> f64 {
        let next_q_values: &Array1<f64> = self.policy.get(&next_obs).unwrap_or(&self.default_value);

        // Sarsa
        let future_q_value: f64 = next_q_values[next_action];

        let curr_q_values: &Array1<f64> = self.policy.get(&curr_obs).unwrap_or(&self.default_value);
        let temporal_difference: f64 =
            reward + self.discount_factor * future_q_value - curr_q_values[curr_action];

        self.policy
            .entry(curr_obs)
            .or_insert(self.default_value.clone())[curr_action] +=
            self.learning_rate * temporal_difference;

        if terminated {
            let new_epsilon: f64 = self.epsilon - self.epsilon_decay;
            self.epsilon = if self.final_epsilon > new_epsilon {
                self.epsilon
            } else {
                new_epsilon
            };
        }
        temporal_difference
    }
    fn get_action(&mut self, obs: usize) -> usize {
        if self.epsilon != 0.0
            && self.exploration_decider.sample(&mut rand::thread_rng()) < self.epsilon
        {
            self.rand_action_selecter.sample(&mut rand::thread_rng())
        } else {
            let values = self.policy.get(&obs).unwrap_or(&self.default_value);
            values
                .iter()
                .enumerate()
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

    fn reset(&mut self) {
        self.policy.clear();
        self.epsilon = self.initial_epsilon;
    }
}
