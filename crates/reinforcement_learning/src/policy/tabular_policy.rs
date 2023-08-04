use environments::env::DiscreteAction;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::IndexMut;

use super::DiscretePolicy;

#[derive(Debug, Clone)]
pub struct TabularPolicy<T: Hash + PartialEq + Eq + Clone, A: DiscreteAction>
where
    [(); A::RANGE]: Sized,
{
    learning_rate: f64,
    default: [f64; A::RANGE],
    policy: HashMap<T, [f64; A::RANGE]>,
}

impl<T: Hash + PartialEq + Eq + Clone, A: DiscreteAction> TabularPolicy<T, A>
where
    [(); A::RANGE]: Sized,
{
    pub fn new(learning_rate: f64, default_value: f64) -> Self {
        Self {
            learning_rate,
            default: [default_value; A::RANGE],
            policy: HashMap::default(),
        }
    }
}

impl<T: Hash + PartialEq + Eq + Clone, A: DiscreteAction> DiscretePolicy<T, A>
    for TabularPolicy<T, A>
where
    [f64]: IndexMut<A, Output = f64>,
    [(); A::RANGE]: Sized,
{
    fn predict(&mut self, obs: &T) -> [f64; A::RANGE] {
        *self.policy.get(obs).unwrap_or(&self.default)
    }

    fn get_values(&mut self, obs: &T) -> [f64; A::RANGE] {
        *self.policy.get(obs).unwrap_or(&self.default)
    }

    fn update(&mut self, obs: &T, action: A, _next_obs: &T, temporal_difference: f64) -> f64 {
        self.policy.entry(obs.clone()).or_insert(self.default)[action] +=
            self.learning_rate * temporal_difference;
        self.learning_rate * temporal_difference
    }

    fn reset(&mut self) {
        self.policy = HashMap::default()
    }

    fn after_update(&mut self) {}
}
