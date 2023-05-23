mod epsilon_greed;

use crate::env::{Observation, ActionSpace};

pub use epsilon_greed::EpsilonGreed;

use crate::Policy;

pub trait ActionSelection  {
    fn prepare_agent(&mut self);
    fn get_action(&self, obs: &Observation, action_space: &ActionSpace, policy: &Policy) -> usize;
    fn update(&mut self);
    fn get_exploration_probs(&self, action_space: &ActionSpace) -> Vec<f64>;
    fn get_exploration_rate(&self) -> f64;
}