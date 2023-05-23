use std::cell::{RefCell, RefMut};

use super::PolicyUpdate;

use crate::{env::{Observation, ActionSpace}, Policy, algorithms::action_selection::ActionSelection};

pub struct SarsaLambda {
    learning_rate: f64,
    discount_factor: f64,
    pub trace: RefCell<Policy>,
    lambda_factor: f64
}

impl SarsaLambda {
    pub fn new(
        learning_rate: f64,
        discount_factor: f64,
        lambda_factor: f64,
        default_value: f64,
        action_space: ActionSpace
    ) -> Self {
        return Self{
            learning_rate,
            discount_factor,
            trace: RefCell::new(Policy::new(default_value, action_space)),
            lambda_factor
        }
    }
}

impl PolicyUpdate for SarsaLambda {
    fn update(
        &mut self,
        curr_obs: Observation,
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
        let values: &mut Vec<f64> = policy.get_mut(curr_obs.clone());
        let temporal_difference: f64 = reward + self.discount_factor * future_q_value - values[curr_action];
        
        let mut trace: RefMut<Policy> = self.trace.borrow_mut();
        trace.get_mut(curr_obs)[curr_action] += 1.0;

        for (obs, values) in &mut policy.values {
            let e_obs_values: &mut Vec<f64> = trace.get_mut(obs.clone());
            for i in 0..values.len() {
                values[i] = values[i] + self.learning_rate * temporal_difference * e_obs_values[i];
                e_obs_values[i] = self.discount_factor * self.lambda_factor * e_obs_values[i]
            }
        }
    }
}