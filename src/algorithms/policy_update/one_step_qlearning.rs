use std::cell::{RefCell, RefMut};
use std::hash::Hash;
use std::rc::Rc;

use super::PolicyUpdate;

use crate::observation::Observation;
use crate::{policy::Policy, utils::argmax, algorithms::action_selection::ActionSelection};

pub struct OneStepQLearning {
    learning_rate: f64,
    discount_factor: f64
}

impl OneStepQLearning {
    pub fn new(learning_rate: f64, discount_factor: f64) -> Self {
        return Self{learning_rate, discount_factor}
    }
}

impl PolicyUpdate for OneStepQLearning {
    fn update(
        &mut self,
        curr_obs: Rc<dyn Observation>,
        curr_action: usize,
        next_obs: Rc<dyn Observation>,
        _next_action: usize,
        reward: f64,
        _terminated: bool,
        mut policy: RefMut<'_, &mut dyn Policy>,
        _action_selection: &Box<RefCell<&mut dyn ActionSelection>>
    ) -> f64 {
        let next_q_values: &Vec<f64> = policy.get_ref(next_obs.clone());
        let future_q_value: f64 = next_q_values[argmax(&next_q_values)];
        let values: &mut Vec<f64> = policy.get_mut(curr_obs);
        let temporal_difference: f64 = reward + self.discount_factor * future_q_value - values[curr_action];
        values[curr_action] = values[curr_action] + self.learning_rate * temporal_difference;
        return temporal_difference;
    }
}

