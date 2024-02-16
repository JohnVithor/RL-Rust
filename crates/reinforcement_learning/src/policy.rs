mod neural_policy;
mod tabular_policy;

use std::collections::HashMap;

use environments::env::DiscreteAction;
// pub use neural_policy::DiscreteNeuralPolicy;
pub use tabular_policy::TabularPolicy;

pub trait DiscretePolicy<T, A: DiscreteAction> {
    fn get_estimed_transitions(&self) -> HashMap<(T, T), [f32; A::RANGE]>;
    fn predict(&mut self, obs: &T) -> [f32; A::RANGE];

    fn get_values(&mut self, obs: &T) -> [f32; A::RANGE];

    fn update(&mut self, obs: &T, action: A, next_obs: &T, temporal_difference: f32) -> f32;

    fn reset(&mut self);

    fn after_update(&mut self);
}
