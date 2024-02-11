use std::collections::HashMap;

use ndarray::{Array, Array1};

use crate::{action_selection::ActionSelection, agent::DiscreteAgent};

pub struct OneStepQlearning {
    action_selection: Box<dyn ActionSelection>,
    learning_rate: f64,
    default_value: Array1<f64>,
    discount_factor: f64,
    policy: HashMap<usize, Array1<f64>>,
}

impl OneStepQlearning {
    pub fn new(
        n_actions: usize,
        action_selection: Box<dyn ActionSelection>,
        learning_rate: f64,
        default_value: f64,
        discount_factor: f64,
    ) -> Self {
        Self {
            action_selection,
            learning_rate,
            default_value: Array::from_elem((n_actions,), default_value),
            discount_factor,
            policy: HashMap::new(),
        }
    }
}

impl DiscreteAgent for OneStepQlearning {
    fn update(
        &mut self,
        curr_obs: usize,
        curr_action: usize,
        reward: f64,
        terminated: bool,
        next_obs: usize,
        _next_action: usize,
    ) -> f64 {
        let next_q_values: &Array1<f64> = self.policy.get(&next_obs).unwrap_or(&self.default_value);

        // Qlearning
        let future_q_value: f64 = next_q_values
            .iter()
            .skip(1)
            .fold(next_q_values[0], |acc, x| acc.max(*x));

        let curr_q_values: &Array1<f64> = self.policy.get(&curr_obs).unwrap_or(&self.default_value);
        let temporal_difference: f64 =
            reward + self.discount_factor * future_q_value - curr_q_values[curr_action];

        self.policy
            .entry(curr_obs)
            .or_insert(self.default_value.clone())[curr_action] +=
            self.learning_rate * temporal_difference;

        if terminated {
            self.action_selection.update();
        }
        temporal_difference
    }

    fn get_action(&mut self, obs: usize) -> usize {
        self.action_selection
            .get_action(obs, self.policy.get(&obs).unwrap_or(&self.default_value))
    }

    fn reset(&mut self) {
        self.policy.clear();
        self.action_selection.reset();
    }
}
