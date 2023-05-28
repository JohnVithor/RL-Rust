use std::cell::RefCell;
use std::hash::Hash;

use super::PolicyUpdate;

use crate::{Policy, utils::argmax, algorithms::action_selection::ActionSelection};

pub struct QStep {
    learning_rate: f64,
    discount_factor: f64
}

impl QStep {
    pub fn new(learning_rate: f64, discount_factor: f64) -> Self {
        return Self{learning_rate, discount_factor}
    }
}

impl<T: Hash+PartialEq+Eq+Clone> PolicyUpdate<T> for QStep {
    fn update(
        &mut self,
        curr_obs: T,
        curr_action: usize,
        next_obs: T,
        _next_action: usize,
        reward: f64,
        _terminated: bool,
        policy: &mut Policy<T>,
        _action_selection: &Box<RefCell<&mut dyn ActionSelection<T>>>
    ) -> f64 {
        let next_q_values: &Vec<f64> = policy.get_ref(next_obs.clone());
        let future_q_value: f64 = next_q_values[argmax(&next_q_values)];
        let values: &mut Vec<f64> = policy.get_mut(curr_obs);
        let temporal_difference: f64 = reward + self.discount_factor * future_q_value - values[curr_action];
        values[curr_action] = values[curr_action] + self.learning_rate * temporal_difference;
        return temporal_difference;
    }
}

