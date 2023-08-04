mod uniform_epsilon_greed;
mod upper_confidence_bound;

use environments::env::DiscreteAction;

pub use uniform_epsilon_greed::UniformEpsilonGreed;
pub use upper_confidence_bound::UpperConfidenceBound;

pub trait ActionSelection<T: Clone, A: DiscreteAction> {
    fn get_action(&mut self, obs: &T, values: &[f64; A::RANGE]) -> A;
    fn update(&mut self);
    fn get_exploration_probs(&mut self, obs: &T, values: &[f64; A::RANGE]) -> [f64; A::RANGE];
    fn reset(&mut self);
}
