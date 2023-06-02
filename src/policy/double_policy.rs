use std::{hash::Hash, rc::Rc};
use fxhash::FxHashMap;

use crate::{env::{ActionSpace}, observation::Observation};

use super::Policy;

pub struct DoublePolicy {
    default: Vec<f64>,
    alpha_values: FxHashMap<Rc<dyn Observation>, Vec<f64>>,
    beta_values: FxHashMap<Rc<dyn Observation>, Vec<f64>>,
    state: bool,
    action_space: ActionSpace,
    temp: Vec<f64>
}

impl DoublePolicy {
    pub fn new(default_value: f64, action_space: ActionSpace) -> Self {
        return Self {
            default: vec![default_value; action_space.size],
            alpha_values: FxHashMap::default(),
            beta_values: FxHashMap::default(),
            state: true, 
            action_space,
            temp: vec![]
        };
    }
}

impl Policy for DoublePolicy {
    fn get_ref(&mut self, curr_obs: Rc<dyn Observation>) -> &Vec<f64> {
        if self.state {
            return self.alpha_values.entry(curr_obs).or_insert(self.default.clone());
        } else {
            return self.beta_values.entry(curr_obs).or_insert(self.default.clone());
        }
    }

    fn get_ref_if_has(&mut self, curr_obs: &Rc<dyn Observation>) -> Option<&Vec<f64>> {
        
        let a_values = self.alpha_values.get(curr_obs);
        let b_values = self.beta_values.get(curr_obs);
        if a_values.is_none() && b_values.is_none() {
            return None;
        }
        if a_values.is_some() {
            self.temp = a_values.unwrap().clone();
            for (i,v) in b_values.unwrap_or(&self.default).iter().enumerate() {
                self.temp[i] += v;
                self.temp[i] /= 2.0;
            }
        } else {
            self.temp = b_values.unwrap().clone();
            for (i,v) in b_values.unwrap_or(&self.default).iter().enumerate() {
                self.temp[i] += v;
                self.temp[i] /= 2.0;
            }
        }

        return Some(&self.temp);
    }

    fn get_mut(&mut self, curr_obs: Rc<dyn Observation>) -> &mut Vec<f64> {
        if self.state {
            return self.beta_values.entry(curr_obs).or_insert(self.default.clone());
        } else {
            return self.alpha_values.entry(curr_obs).or_insert(self.default.clone());
        }
    }

    fn get_mut_if_has(&mut self, curr_obs: &Rc<dyn Observation>) -> Option<&mut Vec<f64>> {
        if self.state {
            return self.beta_values.get_mut(curr_obs);
        } else {
            return self.alpha_values.get_mut(curr_obs);
        }
    }

    fn get_mut_values(&mut self) -> &mut FxHashMap<Rc<dyn Observation>, Vec<f64>>{
        if self.state {
            return &mut self.beta_values;
        } else {
            return &mut self.alpha_values;
        }
    }

    fn get_action_space(&self) -> &ActionSpace {
        return &self.action_space;
    }

    fn reset(&mut self) {
        self.alpha_values = FxHashMap::default();
        self.beta_values = FxHashMap::default();
    }

    fn after_update(&mut self) {
        self.state = !self.state;
    }
}
