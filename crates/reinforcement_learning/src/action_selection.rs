pub mod epsilon_greedy;
mod upper_confidence_bound;

use ndarray::Array1;
pub use upper_confidence_bound::UpperConfidenceBound;

pub trait DiscreteObsDiscreteActionSelection {
    fn get_action(&mut self, obs: usize, values: &Array1<f32>) -> usize;
    fn get_exploration_probs(&mut self, obs: usize, values: &Array1<f32>) -> Array1<f32>;
    fn update(&mut self, epi_reward: f32);
    fn reset(&mut self);
}

pub trait ContinuousObsDiscreteActionSelection {
    fn get_action(&mut self, values: &Array1<f32>) -> usize;
    fn get_exploration_probs(&mut self, values: &Array1<f32>) -> Array1<f32>;
    fn update(&mut self, epi_reward: f32);
    fn reset(&mut self);
}
