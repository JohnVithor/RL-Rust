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
    learning_rate: f32,
    default: [f32; A::RANGE],
    policy: HashMap<T, [f32; A::RANGE]>,
    pub state_changes_counter: HashMap<(T, T), f32>,
    pub state_action_change_counter: HashMap<(T, T), [f32; A::RANGE]>,
}

impl<T: Hash + PartialEq + Eq + Clone, A: DiscreteAction> TabularPolicy<T, A>
where
    [(); A::RANGE]: Sized,
{
    pub fn new(learning_rate: f32, default_value: f32) -> Self {
        Self {
            learning_rate,
            default: [default_value; A::RANGE],
            policy: HashMap::default(),
            state_changes_counter: HashMap::default(),
            state_action_change_counter: HashMap::default(),
        }
    }
}

impl<T: Hash + PartialEq + Eq + Clone, A: DiscreteAction> DiscretePolicy<T, A>
    for TabularPolicy<T, A>
where
    [f32]: IndexMut<A, Output = f32>,
    [(); A::RANGE]: Sized,
{
    fn get_estimed_transitions(&self) -> HashMap<(T, T), [f32; A::RANGE]> {
        let mut result: HashMap<(T, T), [f32; A::RANGE]> = self.state_action_change_counter.clone();
        for (k, v) in result.iter_mut() {
            let c = self.state_changes_counter.get(k).unwrap();
            for x in v.iter_mut() {
                *x /= *c;
            }
        }
        result
    }

    fn predict(&mut self, obs: &T) -> [f32; A::RANGE] {
        *self.policy.get(obs).unwrap_or(&self.default)
    }

    fn get_values(&mut self, obs: &T) -> [f32; A::RANGE] {
        *self.policy.get(obs).unwrap_or(&self.default)
    }

    fn update(&mut self, obs: &T, action: A, next_obs: &T, temporal_difference: f32) -> f32 {
        self.policy.entry(obs.clone()).or_insert(self.default)[action.clone()] +=
            self.learning_rate * temporal_difference;

        *self
            .state_changes_counter
            .entry((obs.clone(), next_obs.clone()))
            .or_insert(0.0) += 1.0;
        self.state_action_change_counter
            .entry((obs.clone(), next_obs.clone()))
            .or_insert([0.0; A::RANGE])[action] += 1.0;

        self.learning_rate * temporal_difference
    }

    fn reset(&mut self) {
        self.policy = HashMap::default();
        self.state_changes_counter = HashMap::default();
        self.state_action_change_counter = HashMap::default();
    }

    fn after_update(&mut self) {}
}
