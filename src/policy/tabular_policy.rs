use fxhash::FxHashMap;
use std::fmt::Debug;
use std::hash::Hash;

use super::Policy;

#[derive(Debug, Clone)]
pub struct TabularPolicy<T: Hash + PartialEq + Eq + Clone, const COUNT: usize> {
    learning_rate: f64,
    default: [f64; COUNT],
    policy: FxHashMap<T, [f64; COUNT]>,
}

impl<T: Hash + PartialEq + Eq + Clone, const COUNT: usize> TabularPolicy<T, COUNT> {
    pub fn new(learning_rate: f64, default_value: f64) -> Self {
        Self {
            learning_rate,
            default: [default_value; COUNT],
            policy: FxHashMap::default(),
        }
    }
}

impl<T: Hash + PartialEq + Eq + Clone + Debug, const COUNT: usize> Policy<T, COUNT>
    for TabularPolicy<T, COUNT>
{
    fn predict(&mut self, obs: &T) -> [f64; COUNT] {
        *self.policy.get(obs).unwrap_or(&self.default)
    }

    fn get_values(&mut self, obs: &T) -> [f64; COUNT] {
        *self.policy.get(obs).unwrap_or(&self.default)
    }

    fn update(&mut self, obs: &T, action: usize, _next_obs: &T, temporal_difference: f64) -> f64 {
        self.policy.entry(obs.clone()).or_insert(self.default)[action] +=
            self.learning_rate * temporal_difference;
        self.learning_rate * temporal_difference
    }

    fn reset(&mut self) {
        self.policy = FxHashMap::default()
    }

    fn after_update(&mut self) {}
}
