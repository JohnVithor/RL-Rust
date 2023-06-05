mod uniform_epsilon_greed;
mod upper_confidence_bound;

use std::hash::Hash;
use enum_dispatch::enum_dispatch;
pub use uniform_epsilon_greed::UniformEpsilonGreed;
pub use upper_confidence_bound::UpperConfidenceBound;

#[enum_dispatch]
pub trait ActionSelection<T: Hash + PartialEq + Eq + Clone, const COUNT: usize> {
    fn get_action(&mut self, obs: &T, values: &[f64; COUNT]) -> usize;
    fn update(&mut self);
    fn get_exploration_probs(&mut self, obs: &T, values: &[f64; COUNT]) -> [f64; COUNT];
    fn reset(&mut self);
}

#[derive(Debug, Clone)]
#[enum_dispatch(ActionSelection<T, COUNT>)]
pub enum EnumActionSelection<T: Hash + PartialEq + Eq + Clone, const COUNT: usize> {
    UniformEpsilonGreed(UniformEpsilonGreed<COUNT>),
    UpperConfidenceBound(UpperConfidenceBound<T, COUNT>),
}

// impl<T: Hash + PartialEq + Eq + Clone, const COUNT: usize> ActionSelection<T, COUNT>
//     for EnumActionSelection<T, COUNT>
// {
//     fn get_action(&mut self, obs: &T, values: &[f64; COUNT]) -> usize {
//         match self {
//             EnumActionSelection::UniformEpsilonGreed(uniform_epsilon_greed) => {
//                 uniform_epsilon_greed.get_action(obs, values)
//             }
//             EnumActionSelection::UpperConfidenceBound(upper_confidence_bound) => {
//                 upper_confidence_bound.get_action(obs, values)
//             }
//         }
//     }

//     fn update(&mut self) {
//         match self {
//             EnumActionSelection::UniformEpsilonGreed(uniform_epsilon_greed) => {
//                 <UniformEpsilonGreed<COUNT> as ActionSelection<T, COUNT>>::update(uniform_epsilon_greed)
//             }
//             EnumActionSelection::UpperConfidenceBound(upper_confidence_bound) => {
//                 upper_confidence_bound.update()
//             }
//         }
//     }

//     fn get_exploration_probs(&mut self, obs: &T, values: &[f64; COUNT]) -> [f64; COUNT] {
//         match self {
//             EnumActionSelection::UniformEpsilonGreed(uniform_epsilon_greed) => {
//                 uniform_epsilon_greed.get_exploration_probs(obs, values)
//             }
//             EnumActionSelection::UpperConfidenceBound(upper_confidence_bound) => {
//                 upper_confidence_bound.get_exploration_probs(obs, values)
//             }
//         }
//     }

//     fn reset(&mut self) {
//         match self {
//             EnumActionSelection::UniformEpsilonGreed(uniform_epsilon_greed) => {
//                 <UniformEpsilonGreed<COUNT> as ActionSelection<T, COUNT>>::reset(uniform_epsilon_greed)
//             }
//             EnumActionSelection::UpperConfidenceBound(upper_confidence_bound) => {
//                 upper_confidence_bound.reset()
//             }
//         }
//     }
// }
