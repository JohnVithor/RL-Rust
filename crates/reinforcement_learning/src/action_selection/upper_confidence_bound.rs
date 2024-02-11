// use environments::env::DiscreteAction;
// use std::{collections::HashMap, hash::Hash};

// use crate::utils::argmax;

// use super::ActionSelection;

// #[derive(Debug, Clone)]
// pub struct UpperConfidenceBound<T: Hash + PartialEq + Eq + Clone, A: DiscreteAction>
// where
//     [(); A::RANGE]: Sized,
// {
//     action_counter: HashMap<T, [u128; A::RANGE]>,
//     t: u128,
//     confidence_level: f64,
// }

// impl<T: Hash + PartialEq + Eq + Clone, A: DiscreteAction> UpperConfidenceBound<T, A>
// where
//     [(); A::RANGE]: Sized,
// {
//     pub fn new(confidence_level: f64) -> Self {
//         Self {
//             action_counter: HashMap::default(),
//             t: 1,
//             confidence_level,
//         }
//     }
// }

// impl<T: Hash + PartialEq + Eq + Clone, A: DiscreteAction> ActionSelection<T, A>
//     for UpperConfidenceBound<T, A>
// where
//     [(); A::RANGE]: Sized,
// {
//     fn get_action(&mut self, obs: &T, values: &[f64; A::RANGE]) -> A {
//         let obs_actions: &mut [u128] = self
//             .action_counter
//             .entry(obs.clone())
//             .or_insert([0; A::RANGE]);
//         let mut ucbs: [f64; A::RANGE] = [0.0; A::RANGE];
//         for i in 0..A::RANGE {
//             ucbs[i] = values[i]
//                 + self.confidence_level
//                     * ((self.t as f64).ln() / (obs_actions[i] as f64 + f64::MIN_POSITIVE)).sqrt()
//         }
//         let action = argmax(&ucbs);
//         obs_actions[action] += 1;
//         self.t += 1;
//         action.into()
//     }

//     fn update(&mut self) {}

//     fn get_exploration_probs(&mut self, obs: &T, values: &[f64; A::RANGE]) -> [f64; A::RANGE] {
//         let obs_actions: &mut [u128] = self
//             .action_counter
//             .entry(obs.clone())
//             .or_insert([0; A::RANGE]);
//         let mut ucbs: [f64; A::RANGE] = [0.0; A::RANGE];
//         for i in 0..A::RANGE {
//             ucbs[i] = values[i]
//                 + self.confidence_level
//                     * ((self.t as f64).ln() / (obs_actions[i] as f64 + f64::MIN_POSITIVE)).sqrt()
//         }
//         let action = argmax(&ucbs);
//         let mut probs: [f64; A::RANGE] = [0.0; A::RANGE];
//         probs[action] = 1.0;
//         probs
//     }

//     fn reset(&mut self) {
//         self.action_counter = HashMap::default();
//         self.t = 1;
//     }
// }
