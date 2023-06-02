use std::cell::{RefCell, RefMut};
use std::hash::Hash;

use super::PolicyUpdate;

use crate::policy::BasicPolicy;
use crate::{env::ActionSpace, policy::Policy, utils::argmax, algorithms::action_selection::ActionSelection};

pub struct QLearningLambda<T: Hash+PartialEq+Eq+Clone> {
    learning_rate: f64,
    discount_factor: f64,
    pub trace: RefCell<BasicPolicy<T>>,
    lambda_factor: f64
}

impl<T: Hash+PartialEq+Eq+Clone> QLearningLambda<T> {
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
            trace: RefCell::new(BasicPolicy::new(default_value, action_space)),
            lambda_factor
        }
    }
}

impl<T: Hash+PartialEq+Eq+Clone> PolicyUpdate<T> for QLearningLambda<T> {
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
        let next_q_values: &Vec<f64> = policy.get_ref(next_obs);
        let best_next_action: usize = argmax(&next_q_values);
        let future_q_value: f64 = next_q_values[best_next_action];
        let values: &mut Vec<f64> = policy.get_mut(curr_obs.clone());
        let temporal_difference: f64 = reward + self.discount_factor * future_q_value - values[curr_action];
        
        let mut trace: RefMut<BasicPolicy<T>> = self.trace.borrow_mut();
        trace.get_mut(curr_obs.clone())[curr_action] += 1.0;

        for (obs, values) in policy.get_mut_values() {
            let t_values: &mut Vec<f64> = trace.get_mut(obs.clone());
            for i in 0..values.len() {
                values[i] = values[i] + self.learning_rate * temporal_difference * t_values[i];
                if next_action == best_next_action {
                    t_values[i] = self.discount_factor * self.lambda_factor * t_values[i]
                } else {
                    t_values[i] = 0.0;
                }
            }
        }
        return temporal_difference;
    }
}