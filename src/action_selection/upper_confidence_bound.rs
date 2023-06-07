use std::hash::Hash;

use fxhash::FxHashMap;

use crate::utils::argmax;

use super::ActionSelection;

#[derive(Debug, Clone)]
pub struct UpperConfidenceBound<T: Hash + PartialEq + Eq + Clone, const COUNT: usize> {
    action_counter: FxHashMap<T, [u128; COUNT]>,
    t: u128,
    confidence_level: f64,
}

impl<T: Hash + PartialEq + Eq + Clone, const COUNT: usize> UpperConfidenceBound<T, COUNT> {
    pub fn new(confidence_level: f64) -> Self {
        Self {
            action_counter: FxHashMap::default(),
            t: 1,
            confidence_level,
        }
    }
}

impl<T: Hash + PartialEq + Eq + Clone, const COUNT: usize> ActionSelection<T, COUNT>
    for UpperConfidenceBound<T, COUNT>
{
    fn get_action(&mut self, obs: &T, values: &[f64; COUNT]) -> usize {
        let obs_actions: &mut [u128; COUNT] =
            self.action_counter.entry(obs.clone()).or_insert([0; COUNT]);
        let mut ucbs: [f64; COUNT] = [0.0; COUNT];
        for i in 0..COUNT {
            ucbs[i] = values[i]
                + self.confidence_level
                    * ((self.t as f64).ln() / (obs_actions[i] as f64 + f64::MIN_POSITIVE)).sqrt()
        }
        let action = argmax(&ucbs);
        obs_actions[action] += 1;
        self.t += 1;
        action
    }

    fn update(&mut self) {
        
    }

    fn get_exploration_probs(&mut self, obs: &T, values: &[f64; COUNT]) -> [f64; COUNT] {
        let obs_actions: &mut [u128; COUNT] =
            self.action_counter.entry(obs.clone()).or_insert([0; COUNT]);
        let mut ucbs: [f64; COUNT] = [0.0; COUNT];
        let mut sum = 0.0;
        for i in 0..COUNT {
            ucbs[i] = values[i]
                + self.confidence_level
                    * ((self.t as f64).ln() / (obs_actions[i] as f64 + f64::MIN_POSITIVE)).sqrt();
            sum += ucbs[i];
        }
        for v in &mut ucbs {
            *v /= sum;
        }
        ucbs
    }

    fn reset(&mut self) {
        self.action_counter = FxHashMap::default();
        self.t = 1;
    }
}
