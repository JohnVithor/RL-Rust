use std::hash::Hash;

use fxhash::FxHashMap;

use super::Policy;

#[derive(Debug, Clone)]
pub struct DoubleTabularPolicy<T: Hash + PartialEq + Eq + Clone, const COUNT: usize> {
    default: [f64; COUNT],
    alpha_policy: FxHashMap<T, [f64; COUNT]>,
    beta_policy: FxHashMap<T, [f64; COUNT]>,
    policy_flag: bool,
}

impl<T: Hash + PartialEq + Eq + Clone, const COUNT: usize> Policy<T, COUNT>
    for DoubleTabularPolicy<T, COUNT>
{
    fn predict(&mut self, obs: &T) -> [f64; COUNT] {
        let mut values: [f64; COUNT] = [0.0; COUNT];
        let a_values: &[f64; COUNT] = self.alpha_policy.get(obs).unwrap_or(&self.default);
        let b_values: &[f64; COUNT] = self.beta_policy.get(obs).unwrap_or(&self.default);
        for i in 0..COUNT {
            values[i] = a_values[i] + b_values[i];
        }
        values
    }

    fn get_values(&mut self, obs: &T) -> &[f64; COUNT] {
        match self.policy_flag {
            true => &self.alpha_policy,
            false => &self.beta_policy,
        }
        .get(obs)
        .unwrap_or(&self.default)
    }

    fn update(&mut self, obs: &T, action: usize, value: f64) {
        match self.policy_flag {
            true => &mut self.beta_policy,
            false => &mut self.alpha_policy,
        }
        .entry(obs.clone())
        .or_insert(self.default)[action] += value;
    }

    fn reset(&mut self) {
        self.alpha_policy = FxHashMap::default();
        self.beta_policy = FxHashMap::default();
    }

    fn after_update(&mut self) {
        self.policy_flag = !self.policy_flag;
    }
}
