mod uniform_epsilon_greed;
mod upper_confidence_bound;

pub use uniform_epsilon_greed::UniformEpsilonGreed;
pub use upper_confidence_bound::UpperConfidenceBound;

pub trait ActionSelection<T: Clone, const COUNT: usize> {
    fn get_action(&mut self, obs: &T, values: &[f64; COUNT]) -> usize;
    fn update(&mut self);
    fn get_exploration_probs(&mut self, obs: &T, values: &[f64; COUNT]) -> [f64; COUNT];
    fn reset(&mut self);
}
