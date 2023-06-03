use std::hash::Hash;
use fxhash::FxHashMap;

use crate::env::ActionSpace;

use super::Policy;
#[derive(Debug, Clone)]
pub struct DoublePolicy<T> {
    learning_rate: f64,
    discount_factor: f64,
    default: ndarray::Array1<f64>,
    alpha_values: FxHashMap<T, ndarray::Array1<f64>>,
    beta_values: FxHashMap<T, ndarray::Array1<f64>>,
    state: bool,
    action_space: ActionSpace,
    temp: ndarray::Array1<f64>
}

impl<T: Hash+PartialEq+Eq+Clone> DoublePolicy<T> {
    pub fn new(learning_rate: f64, discount_factor:f64, default_value: f64, action_space: ActionSpace) -> Self {
        return Self {
            default: ndarray::Array::from_elem(action_space.size, default_value),
            alpha_values: FxHashMap::default(),
            beta_values: FxHashMap::default(),
            state: true, 
            temp: ndarray::Array1::default(action_space.size),
            action_space,
            learning_rate,
            discount_factor
        };
    }
}

impl<T: Hash+PartialEq+Eq+Clone> Policy<T> for DoublePolicy<T>{
    fn get_values(&mut self, curr_obs: T) -> &ndarray::Array1<f64> {
        if self.state {
            return self.alpha_values.entry(curr_obs).or_insert(self.default.clone());
        } else {
            return self.beta_values.entry(curr_obs).or_insert(self.default.clone());
        }
    }

    fn update_values(&mut self, curr_obs: T, curr_action: usize, _next_obs: T, _next_action: usize, temporal_difference: f64) {
        let values: &mut ndarray::Array1<f64>;
        if self.state {
            values = self.beta_values.entry(curr_obs).or_insert(self.default.clone());
        } else {
            values = self.alpha_values.entry(curr_obs).or_insert(self.default.clone());
        }
        values[curr_action] = values[curr_action] + self.learning_rate * temporal_difference;
    }

    fn predict(&mut self, curr_obs: T) -> &ndarray::Array1<f64> {
        let a_values = self.alpha_values.get(&curr_obs);
        let b_values = self.beta_values.get(&curr_obs);
        if a_values.is_none() && b_values.is_none() {
            self.alpha_values.entry(curr_obs.clone()).or_insert(self.default.clone());
            self.beta_values.entry(curr_obs).or_insert(self.default.clone());
            return &self.default;
        }
        if a_values.is_some() {
            self.temp = a_values.unwrap().clone();
            for (i,v) in b_values.unwrap_or(&self.default).iter().enumerate() {
                self.temp[i] += v;
                self.temp[i] /= 2.0;
            }
        } else {
            self.temp = b_values.unwrap().clone();
            for (i,v) in b_values.unwrap_or(&self.default).iter().enumerate() {
                self.temp[i] += v;
                self.temp[i] /= 2.0;
            }
        }

        return &self.temp;
    }

    fn get_action_space(&self) -> &ActionSpace {
        return &self.action_space;
    }

    fn reset(&mut self) {
        self.alpha_values = FxHashMap::default();
        self.beta_values = FxHashMap::default();
    }

    fn after_update(&mut self) {
        self.state = !self.state;
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
        let values: &ndarray::Array1<f64>;
        if self.state {
            values = self.beta_values.entry(curr_obs).or_insert(self.default.clone());
        } else {
            values = self.alpha_values.entry(curr_obs).or_insert(self.default.clone());
        }
        let temporal_difference: f64 = reward + self.discount_factor * future_q_value - values[curr_action];
        return temporal_difference;
    }
}
