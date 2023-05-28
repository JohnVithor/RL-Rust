use std::cell::RefCell;
use std::hash::Hash;

use super::PolicyUpdate;

use crate::{env::ActionSpace, Policy, algorithms::action_selection::ActionSelection, utils::argmax};

pub struct ExpectedSarsaStep {
    learning_rate: f64,
    discount_factor: f64
}

impl ExpectedSarsaStep {
    pub fn new(learning_rate: f64, discount_factor: f64) -> Self {
        return Self{learning_rate, discount_factor}
    }
}

impl<T: Hash+PartialEq+Eq+Clone> PolicyUpdate<T> for ExpectedSarsaStep {
    fn update(
        &mut self,
        curr_obs: T,
        curr_action: usize,
        next_obs: T,
        _next_action: usize,
        reward: f64,
        _terminated: bool,
        policy: &mut Policy<T>,
        action_selection: &Box<RefCell<dyn ActionSelection<T>>>
    ) -> f64 {
        let action_space: ActionSpace = policy.action_space.clone();
        let next_q_values: &Vec<f64> = policy.get_ref(next_obs.clone());
        let epsilon: f64 = action_selection.borrow().get_exploration_rate();
        let policy_probs: Vec<f64> = action_selection.borrow().get_exploration_probs(&action_space);
        let best_action_value: f64 = next_q_values[argmax(&next_q_values)];
        
        let mut n_max_action: i32 = 0;
        for i in 0..action_space.size {
            if next_q_values[i] == best_action_value {
                n_max_action += 1;
            }
        }
        let mut future_q_value: f64 = 0.0;
        for i in 0..action_space.size {
            if next_q_values[i] == best_action_value {
                future_q_value += (policy_probs[i] + (1.0-epsilon) / n_max_action as f64) * next_q_values[i]
            } else {
                future_q_value += policy_probs[i] * next_q_values[i]
            }
        }
        
        let values: &mut Vec<f64> = policy.get_mut(curr_obs);
        let temporal_difference: f64 = reward + self.discount_factor * future_q_value - values[curr_action];
        values[curr_action] = values[curr_action] + self.learning_rate * temporal_difference;
        return temporal_difference;
    }
}

