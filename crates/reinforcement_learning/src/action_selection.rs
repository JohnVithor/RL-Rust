mod uniform_epsilon_greed;
mod upper_confidence_bound;

use ndarray::Array1;
pub use uniform_epsilon_greed::UniformEpsilonGreed;
// pub use upper_confidence_bound::UpperConfidenceBound;

pub trait ActionSelection {
    fn get_action(&mut self, obs: usize, values: &Array1<f64>) -> usize;
    fn update(&mut self);
    fn get_exploration_probs(&mut self, obs: usize, values: &Array1<f64>) -> Array1<f64>;
    fn reset(&mut self);
}
