mod uniform_epsilon_greed;
mod upper_confidence_bound;

use crate::env::ActionSpace;

pub use uniform_epsilon_greed::UniformEpsilonGreed;
pub use upper_confidence_bound::UpperConfidenceBound;

use crate::Policy;

pub trait ActionSelection<T>  {
    fn get_action(&self, obs: &T, action_space: &ActionSpace, policy: &Policy<T>) -> usize;
    fn update(&mut self);
    fn get_exploration_probs(&self, action_space: &ActionSpace) -> Vec<f64>;
    fn get_exploration_rate(&self) -> f64;
}