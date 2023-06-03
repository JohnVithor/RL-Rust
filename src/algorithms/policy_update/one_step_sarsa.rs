use std::cell::{RefCell, RefMut};
use std::hash::Hash;

use super::PolicyUpdate;

use crate::{policy::Policy, algorithms::action_selection::ActionSelection};

pub struct OneStepSARSA {
    learning_rate: f64,
    discount_factor: f64
}

impl OneStepSARSA {
    pub fn new(learning_rate: f64, discount_factor: f64) -> Self {
        return Self{learning_rate, discount_factor}
    }
}

impl<T: Hash+PartialEq+Eq+Clone> PolicyUpdate<T> for OneStepSARSA {
    fn update(
        &mut self,
        curr_obs: T,
        curr_action: usize,
        next_obs: T,
        next_action: usize,
        reward: f64,
        _terminated: bool,
        mut policy: RefMut<'_, &mut dyn Policy<T>>,
        _action_selection: &Box<RefCell<&mut dyn ActionSelection<T>>>
    ) -> f64 {
        let next_q_values: &Vec<f64> = policy.get_ref(next_obs.clone());
        let future_q_value: f64 = next_q_values[next_action];
        let values: &mut Vec<f64> = policy.get_mut(curr_obs);
        let temporal_difference: f64 = reward + self.discount_factor * future_q_value - values[curr_action];
        values[curr_action] = values[curr_action] + self.learning_rate * temporal_difference;
        return temporal_difference;
    }
}

