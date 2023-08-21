mod neural_policy;
mod tabular_policy;

use environments::env::DiscreteAction;
// pub use neural_policy::DiscreteNeuralPolicy;
pub use tabular_policy::TabularPolicy;

pub trait DiscretePolicy<T, A: DiscreteAction> {
    fn predict(&mut self, obs: &T) -> [f64; A::RANGE];

    fn get_values(&mut self, obs: &T) -> [f64; A::RANGE];

    fn update(&mut self, obs: &T, action: A, next_obs: &T, temporal_difference: f64) -> f64;

    fn reset(&mut self);

    fn after_update(&mut self);
}
