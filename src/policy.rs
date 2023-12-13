use enum_dispatch::enum_dispatch;
use fxhash::FxHashMap;
use std::fmt::Debug;
use std::hash::Hash;
mod tabular_policy;

pub use tabular_policy::TabularPolicy;

#[enum_dispatch]
pub trait Policy<T: Hash + PartialEq + Eq + Clone, const COUNT: usize> {
    fn predict(&mut self, obs: &T) -> [f64; COUNT];

    fn get_values(&mut self, obs: &T) -> [f64; COUNT];

    fn update(&mut self, obs: &T, action: usize, next_obs: &T, temporal_difference: f64) -> f64;

    fn reset(&mut self);

    fn after_update(&mut self);

    fn get_estimed_transitions(&self) -> FxHashMap<(T, T), [f64; COUNT]>;
}

#[enum_dispatch(Policy<T, COUNT>)]
#[derive(Debug)]
pub enum EnumPolicy<T: Hash + PartialEq + Eq + Clone + Debug, const COUNT: usize> {
    TabularPolicy(TabularPolicy<T, COUNT>),
}
