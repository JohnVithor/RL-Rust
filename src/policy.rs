use enum_dispatch::enum_dispatch;
use std::fmt::Debug;
use std::hash::Hash;
mod double_tabular_policy;
mod main_target_neural_policy;
mod neural_policy;
mod tabular_policy;

pub use double_tabular_policy::DoubleTabularPolicy;
pub use main_target_neural_policy::MainTargetNeuralPolicy;
pub use neural_policy::NeuralPolicy;
pub use tabular_policy::TabularPolicy;

#[enum_dispatch]
pub trait Policy<T: Hash + PartialEq + Eq + Clone, const COUNT: usize> {
    fn predict(&mut self, obs: &T) -> [f64; COUNT];

    fn get_values(&mut self, obs: &T) -> [f64; COUNT];

    fn update(&mut self, obs: &T, action: usize, next_obs: &T, temporal_difference: f64) -> f64;

    fn reset(&mut self);

    fn after_update(&mut self);
}

#[enum_dispatch(Policy<T, COUNT>)]
#[derive(Debug)]
pub enum EnumPolicy<T: Hash + PartialEq + Eq + Clone + Debug, const COUNT: usize> {
    TabularPolicy(TabularPolicy<T, COUNT>),
    DoubleTabularPolicy(DoubleTabularPolicy<T, COUNT>),
    NeuralPolicy(NeuralPolicy<T, COUNT>),
}
