use std::cell::RefCell;

use super::PolicyUpdate;

use crate::{env::Observation, Policy, algorithms::action_selection::ActionSelection};

pub struct SarsaStep {
    learning_rate: f64,
    discount_factor: f64
}

impl SarsaStep {
    pub fn new(learning_rate: f64, discount_factor: f64) -> Self {
        return Self{learning_rate, discount_factor}
    }
}

impl PolicyUpdate for SarsaStep {
    fn update(
        &mut self,
        curr_obs:Observation,
        curr_action: usize,
        next_obs: Observation,
        next_action: usize,
        reward: f64,
        _terminated: bool,
        policy: &mut Policy,
        _action_selection: &Box<RefCell<dyn ActionSelection>>
    ) {
        let next_q_values: &Vec<f64> = policy.get_ref(next_obs.clone());
        let future_q_value = next_q_values[next_action];
        let values: &mut Vec<f64> = policy.get_mut(curr_obs);
        let temporal_difference: f64 = reward + self.discount_factor * future_q_value - values[curr_action];
        values[curr_action] = values[curr_action] + self.learning_rate * temporal_difference;
    }
}

