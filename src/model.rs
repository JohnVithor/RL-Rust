mod random_model;

pub use random_model::RandomModel;

use std::fmt::Debug;
use std::hash::Hash;
use enum_dispatch::enum_dispatch;

#[enum_dispatch]
pub trait Model<T: Hash + PartialEq + Eq + Clone + Debug, const COUNT: usize> {
    fn get_info(&self) -> (T, usize, f64, T, usize);
    fn add_info(&mut self, obs: T, action: usize, reward: f64, next_obs: T, next_action: usize);
    fn reset(&mut self);
}

#[enum_dispatch(Model<T, COUNT>)]
#[derive(Debug, Clone)]
pub enum EnumModel<T: Hash + PartialEq + Eq + Clone + Debug, const COUNT: usize> {
    RandomModel(RandomModel<T, COUNT>)
}