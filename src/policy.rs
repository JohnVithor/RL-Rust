mod basic_policy;
mod double_policy;

pub use basic_policy::BasicPolicy;
pub use double_policy::DoublePolicy;
use fxhash::FxHashMap;
use std::hash::Hash;

use crate::env::ActionSpace;
pub trait Policy<T: Hash + PartialEq + Eq + Clone> {
    fn get_action_space(&self) -> &ActionSpace;

    fn get_ref(&mut self, curr_obs: T) -> &Vec<f64>;

    fn get_ref_if_has(&mut self, curr_obs: &T) -> Option<&Vec<f64>>;

    fn get_mut(&mut self, curr_obs: T) -> &mut Vec<f64>;

    fn get_mut_if_has(&mut self, curr_obs: &T) -> Option<&mut Vec<f64>>;

    fn get_mut_values(&mut self) -> &mut FxHashMap<T, Vec<f64>>;

    fn reset(&mut self);

    fn after_update(&mut self);
}
