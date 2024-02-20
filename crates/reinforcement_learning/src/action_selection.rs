mod adaptative_epsilon;
mod epsilon_decreasing;
mod epsilon_greedy;
mod upper_confidence_bound;

pub use adaptative_epsilon::AdaptativeEpsilon;
pub use epsilon_decreasing::EpsilonDecreasing;
pub use epsilon_greedy::EpsilonGreedy;
use ndarray::Array1;
pub use upper_confidence_bound::UpperConfidenceBound;

pub trait DiscreteObsDiscreteActionSelection {
    fn get_action(&mut self, obs: usize, values: &Array1<f32>) -> usize;
    fn update(&mut self, reward: f32);
    fn get_exploration_probs(&mut self, obs: usize, values: &Array1<f32>) -> Array1<f32>;
    fn reset(&mut self);
}

pub trait ContinuousObsDiscreteActionSelection {
    fn get_action(&mut self, values: &Array1<f32>) -> usize;
    fn update(&mut self, reward: f32);
    fn get_exploration_probs(&mut self, values: &Array1<f32>) -> Array1<f32>;
    fn reset(&mut self);
}
