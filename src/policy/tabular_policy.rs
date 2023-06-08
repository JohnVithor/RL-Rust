use std::hash::Hash;

use fxhash::FxHashMap;

use super::Policy;

#[derive(Debug, Clone)]
pub struct TabularPolicy<T: Hash + PartialEq + Eq + Clone, const COUNT: usize> {
    default: [f64; COUNT],
    policy: FxHashMap<T, [f64; COUNT]>,
}

impl<T: Hash + PartialEq + Eq + Clone, const COUNT: usize> TabularPolicy<T, COUNT> {
    pub fn new(default_value: f64) -> Self {
        Self {
            default: [default_value; COUNT],
            policy: FxHashMap::default(),
        }
    }
}

impl<T: Hash + PartialEq + Eq + Clone, const COUNT: usize> Policy<T, COUNT>
    for TabularPolicy<T, COUNT>
{
    fn predict(&mut self, obs: &T) -> [f64; COUNT] {
        *self.policy.get(obs).unwrap_or(&self.default)
    }

    fn get_values(&mut self, obs: &T) -> [f64; COUNT] {
        *self.policy.get(obs).unwrap_or(&self.default)
    }

    fn update(&mut self, obs: &T, action: usize, value: f64) {
        self.policy
            .entry(obs.clone())
            .or_insert(self.default)[action] += value;
    }

    fn reset(&mut self) {
        self.policy = FxHashMap::default()
    }

    fn after_update(&mut self) {}
}
