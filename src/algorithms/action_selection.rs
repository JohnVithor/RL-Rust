mod uniform_epsilon_greed;
mod upper_confidence_bound;

use std::{cell::RefMut, rc::Rc};

use crate::{env::{ActionSpace}, observation::Observation};

pub use uniform_epsilon_greed::UniformEpsilonGreed;
pub use upper_confidence_bound::UpperConfidenceBound;

use crate::policy::Policy;

pub trait ActionSelection  {
    fn get_action(&self, obs: &Rc<dyn Observation>, policy: &mut RefMut<& mut (dyn Policy)>) -> usize;
    fn update(&mut self);
    fn get_exploration_probs(&self, action_space: &ActionSpace) -> Vec<f64>;
    fn get_exploration_rate(&self) -> f64;
    fn reset(&mut self);
}