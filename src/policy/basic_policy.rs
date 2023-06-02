use std::hash::Hash;
use fxhash::FxHashMap;

use crate::env::ActionSpace;

use super::Policy;

pub struct BasicPolicy<T> {
    default: Vec<f64>,
    values: FxHashMap<T, Vec<f64>>,
    action_space: ActionSpace,
}

impl<T: Hash+PartialEq+Eq+Clone> BasicPolicy<T> {
    pub fn new(default_value: f64, action_space: ActionSpace) -> Self {
        return Self {
            default: vec![default_value; action_space.size],
            values: FxHashMap::default(),
            action_space,
        };
    }
}

impl<T: Hash+PartialEq+Eq+Clone> Policy<T> for BasicPolicy<T>{
    fn get_ref(&mut self, curr_obs: T) -> &Vec<f64> {
        return self.values.entry(curr_obs).or_insert(self.default.clone());
    }

    fn get_ref_if_has(&mut self, curr_obs: &T) -> Option<&Vec<f64>> {
        return self.values.get(curr_obs);
    }

    fn get_mut(&mut self, curr_obs: T) -> &mut Vec<f64> {
        return self.values.entry(curr_obs).or_insert(self.default.clone());
    }

    fn get_mut_if_has(&mut self, curr_obs: &T) -> Option<&mut Vec<f64>> {
        return self.values.get_mut(curr_obs);
    }

    fn get_mut_values(&mut self) -> &mut FxHashMap<T, Vec<f64>>{
        return &mut self.values
    }

    fn get_action_space(&self) -> &ActionSpace {
        return &self.action_space;
    }

    fn reset(&mut self) {
        self.values = FxHashMap::default();
    }

    fn after_update(&mut self) {
        
    }
}
