use std::hash::Hash;
use enum_dispatch::enum_dispatch;

mod tabular_policy;
mod double_tabular_policy;
mod neural_policy;

pub use tabular_policy::TabularPolicy;
pub use double_tabular_policy::DoubleTabularPolicy;
pub use neural_policy::NeuralPolicy;
#[enum_dispatch]
pub trait Policy<T: Hash + PartialEq + Eq + Clone, const COUNT: usize> {
    fn predict(&mut self, obs: &T) -> [f64; COUNT];

    fn get_values(&mut self, obs: &T) -> [f64; COUNT];
    
    fn update(&mut self, obs: &T, action: usize, temporal_difference: f64);

    fn reset(&mut self);

    fn after_update(&mut self);
}

#[enum_dispatch(Policy<T, COUNT>)]
#[derive(Debug, Clone)]
pub enum EnumPolicy<T: Hash + PartialEq + Eq + Clone, const COUNT: usize> {
    TabularPolicy(TabularPolicy<T, COUNT>),
    DoubleTabularPolicy(DoubleTabularPolicy<T, COUNT>),
}