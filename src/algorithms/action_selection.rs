mod uniform_epsilon_greed;
mod upper_confidence_bound;

use std::cell::RefMut;

use crate::env::ActionSpace;

pub use uniform_epsilon_greed::UniformEpsilonGreed;
pub use upper_confidence_bound::UpperConfidenceBound;

use crate::policy::Policy;

pub trait ActionSelection<T>  {
    fn get_action(&self, obs: &T, policy: &mut RefMut<& mut (dyn Policy<T>)>) -> usize;
    fn update(&mut self);
    fn get_exploration_probs(&self, action_space: &ActionSpace) -> ndarray::Array1<f64>;
    fn get_exploration_rate(&self) -> f64;
    fn reset(&mut self);
}