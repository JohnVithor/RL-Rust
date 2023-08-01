use std::fmt::Debug;
mod main_target_neural_policy;
mod neural_policy;
mod tabular_policy;

pub use main_target_neural_policy::MainTargetNeuralPolicy;
pub use neural_policy::NeuralPolicy;
pub use tabular_policy::TabularPolicy;

pub trait Policy<T: Clone + Debug, const COUNT: usize> {
    fn predict(&mut self, obs: &T) -> [f64; COUNT];

    fn get_values(&mut self, obs: &T) -> [f64; COUNT];

    fn update(&mut self, obs: &T, action: usize, next_obs: &T, temporal_difference: f64) -> f64;

    fn reset(&mut self);

    fn after_update(&mut self);
}
