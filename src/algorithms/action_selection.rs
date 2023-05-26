mod uniform_epsilon_greed;
mod upper_confidence_bound;

use crate::env::{Observation, ActionSpace};

pub use uniform_epsilon_greed::UniformEpsilonGreed;
pub use upper_confidence_bound::UpperConfidenceBound;

use crate::Policy;

pub trait ActionSelection  {
    fn get_action(&self, obs: &Observation, action_space: &ActionSpace, policy: &Policy) -> usize;
    fn update(&mut self);
    fn get_exploration_probs(&self, action_space: &ActionSpace) -> Vec<f64>;
    fn get_exploration_rate(&self) -> f64;
}