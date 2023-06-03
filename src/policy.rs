mod basic_policy;
mod double_policy;
mod trace_policy;

pub use basic_policy::BasicPolicy;
pub use double_policy::DoublePolicy;
pub use trace_policy::{TracePolicy, TraceMode};

use std::hash::Hash;

use crate::env::ActionSpace;
pub trait Policy<T: Hash+PartialEq+Eq+Clone> {
    fn get_action_space(&self) -> &ActionSpace;

    fn get_values(&mut self, curr_obs: T) -> &ndarray::Array1<f64>;

    fn predict(&mut self, curr_obs: T) -> &ndarray::Array1<f64>;

    fn get_td(&mut self, curr_obs: T, curr_action: usize, reward: f64, future_q_value: f64) -> f64;

    fn update_values(&mut self, curr_obs: T, curr_action: usize, next_obs: T, next_action: usize, temporal_difference: f64);

    fn get_learning_rate(&self) -> f64;

    fn get_discount_factor(&self) -> f64;

    fn get_default(&self) -> &ndarray::Array1<f64>;

    fn reset(&mut self);

    fn after_update(&mut self);
}
