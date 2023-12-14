use std::hash::Hash;

use fxhash::FxHashMap;

use super::Policy;

#[derive(Debug, Clone)]
pub struct TabularPolicy<T: Hash + PartialEq + Eq + Clone, const COUNT: usize> {
    learning_rate: f64,
    default: [f64; COUNT],
    policy: FxHashMap<T, [f64; COUNT]>,
    pub state_changes_counter: FxHashMap<(T, T), f64>,
    pub state_action_change_counter: FxHashMap<(T, T), [f64; COUNT]>,
}

impl<T: Hash + PartialEq + Eq + Clone, const COUNT: usize> TabularPolicy<T, COUNT> {
    pub fn new(learning_rate: f64, default_value: f64) -> Self {
        Self {
            learning_rate,
            default: [default_value; COUNT],
            policy: FxHashMap::default(),
            state_changes_counter: FxHashMap::default(),
            state_action_change_counter: FxHashMap::default(),
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

    fn update(&mut self, obs: &T, action: usize, next_obs: &T, temporal_difference: f64) -> f64 {
        self.policy.entry(obs.clone()).or_insert(self.default)[action] +=
            self.learning_rate * temporal_difference;

        *self
            .state_changes_counter
            .entry((obs.clone(), next_obs.clone()))
            .or_insert(0.0) += 1.0;
        self.state_action_change_counter
            .entry((obs.clone(), next_obs.clone()))
            .or_insert([0.0; COUNT])[action] += 1.0;

        self.learning_rate * temporal_difference
    }

    fn reset(&mut self) {
        self.policy = FxHashMap::default()
    }

    fn after_update(&mut self) {}
    fn get_estimed_transitions(
        &self,
    ) -> (FxHashMap<(T, T), [f64; COUNT]>, &FxHashMap<T, [f64; COUNT]>) {
        let mut result: FxHashMap<(T, T), [f64; COUNT]> = self.state_action_change_counter.clone();
        for (k, v) in result.iter_mut() {
            let c = self.state_changes_counter.get(k).unwrap();
            for x in v.iter_mut() {
                *x /= *c;
            }
        }
        (result, &self.policy)
    }
}
