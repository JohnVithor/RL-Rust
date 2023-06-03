use std::{hash::Hash, array};
use fxhash::FxHashMap;
use ndarray::arr1;

use crate::env::ActionSpace;

use super::Policy;

#[derive(Debug, Clone)]
pub struct BasicPolicy<T> {
    learning_rate: f64,
    discount_factor: f64,
    default: ndarray::Array1<f64>,
    values: FxHashMap<T, ndarray::Array1<f64>>,
    action_space: ActionSpace,
}
impl<T: Hash+PartialEq+Eq+Clone> BasicPolicy<T> {
    pub fn new(learning_rate: f64, discount_factor:f64, default_value: f64, action_space: ActionSpace) -> Self {
        return Self {
            default: ndarray::Array::from_elem(action_space.size, default_value),
            values: FxHashMap::default(),
            action_space,
            learning_rate,
            discount_factor
        };
    }
}

impl<T: Hash+PartialEq+Eq+Clone> Policy<T> for BasicPolicy<T>{
    fn get_values(&mut self, curr_obs: T) -> &ndarray::Array1<f64> {
        return self.values.entry(curr_obs).or_insert(self.default.clone());
    }

    fn predict(&mut self, curr_obs: T) -> &ndarray::Array1<f64> {
        return self.values.entry(curr_obs).or_insert(self.default.clone());
    }

    fn update_values(&mut self, curr_obs: T, curr_action: usize, _next_obs: T, _next_action: usize, temporal_difference: f64) {
        let values: &mut ndarray::Array1<f64> = self.values.entry(curr_obs).or_insert(self.default.clone());
        values[curr_action] = values[curr_action] + self.learning_rate * temporal_difference;
    }

    fn get_action_space(&self) -> &ActionSpace {
        return &self.action_space;
    }

    fn reset(&mut self) {
        self.values = FxHashMap::default();
    }

    fn after_update(&mut self) {
        
    }

    fn get_learning_rate(&self) -> f64 {
        return self.learning_rate;
    }

    fn get_discount_factor(&self) -> f64 {
        return self.discount_factor;
    }

    fn get_default(&self) -> &ndarray::Array1<f64> {
        return &self.default;
    }

    fn get_td(&mut self, curr_obs: T, curr_action: usize, reward: f64, future_q_value: f64) -> f64 {
        let values: &mut ndarray::Array1<f64> = self.values.entry(curr_obs).or_insert(self.default.clone());
        let temporal_difference: f64 = reward + self.discount_factor * future_q_value - values[curr_action];
        return temporal_difference;
    }
}
