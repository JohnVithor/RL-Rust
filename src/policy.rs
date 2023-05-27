use std::collections::HashMap;
use std::hash::Hash;
use crate::env::ActionSpace;

#[derive(Debug, Clone)]
pub struct Policy<T> {
    pub default: Vec<f64>,
    pub values: HashMap<T, Vec<f64>>,
    pub action_space: ActionSpace,
}

impl<T: Hash+PartialEq+Eq+Clone> Policy<T> {
    pub fn new(default_value: f64, action_space: ActionSpace) -> Self {
        return Self {
            default: vec![default_value; action_space.size],
            values: HashMap::new(),
            action_space,
        };
    }

    pub fn get_ref(&mut self, curr_obs: T) -> &Vec<f64> {
        return self.values.entry(curr_obs).or_insert(self.default.clone());
    }

    pub fn get_ref_if_has(&self, curr_obs: &T) -> Option<&Vec<f64>> {
        return self.values.get(curr_obs);
    }

    pub fn get_mut(&mut self, curr_obs: T) -> &mut Vec<f64> {
        return self.values.entry(curr_obs).or_insert(self.default.clone());
    }

    pub fn get_mut_if_has(&mut self, curr_obs: &T) -> Option<&mut Vec<f64>> {
        return self.values.get_mut(curr_obs);
    }
}
