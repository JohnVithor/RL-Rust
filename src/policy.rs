use std::collections::HashMap;

use crate::env::{ActionSpace, Observation};

#[derive(Debug, Clone)]
pub struct Policy {
    pub default: Vec<f64>,
    pub values: HashMap<Observation, Vec<f64>>,
    pub action_space: ActionSpace,
}

impl Policy {
    pub fn new(default_value: f64, action_space: ActionSpace) -> Self {
        return Self {
            default: vec![default_value; action_space.size],
            values: HashMap::new(),
            action_space,
        };
    }

    pub fn get_ref(&mut self, curr_obs: Observation) -> &Vec<f64> {
        return self.values.entry(curr_obs).or_insert(self.default.clone());
    }

    pub fn get_ref_if_has(&self, curr_obs: &Observation) -> Option<&Vec<f64>> {
        return self.values.get(curr_obs);
    }

    pub fn get_mut(&mut self, curr_obs: Observation) -> &mut Vec<f64> {
        return self.values.entry(curr_obs).or_insert(self.default.clone());
    }

    pub fn get_mut_if_has(&mut self, curr_obs: &Observation) -> Option<&mut Vec<f64>> {
        return self.values.get_mut(curr_obs);
    }
}
