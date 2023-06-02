use std::{hash::Hash, cell::{RefCell, RefMut}, rc::Rc};

use fxhash::FxHashMap;

use crate::{env::{ActionSpace}, utils::argmax, policy::Policy, observation::Observation};

use super::ActionSelection;

#[derive(Debug, Clone)]
pub struct UpperConfidenceBound {
    action_counter: RefCell<FxHashMap<Rc<dyn Observation>, Vec<u128>>>,
    t: RefCell<u128>,
    confidence_level: f64
}

impl UpperConfidenceBound {
    pub fn new(confidence_level: f64) -> Self {
        return Self {
            action_counter: RefCell::new(FxHashMap::default()),
            t: RefCell::new(1),
            confidence_level
        }
    }
}

impl ActionSelection for UpperConfidenceBound {
    fn get_action(&self, obs: &Rc<dyn Observation>,  policy: &mut RefMut<&mut dyn Policy>) -> usize {
        let ac = policy.get_action_space().clone();
        match policy.get_ref_if_has(obs) {
            Some(value) => {
            let mut action_counter = self.action_counter.borrow_mut();
            let obs_actions = action_counter.entry(obs.clone()).or_insert(vec![0;ac.size]);
            let mut t: RefMut<u128> = self.t.borrow_mut();
            let ucbs: Vec<f64> = value.iter().enumerate().map(
                |(i, v)| { 
                return v + self.confidence_level * ((*t as f64).ln() / (obs_actions[i] as f64  + f64::MIN_POSITIVE)).sqrt()
            }).collect();
            let action = argmax(&ucbs);
            obs_actions[action] += 1;
            *t += 1;
            return action;
            },
            None => return policy.get_action_space().sample()
        }
    }

    fn update(&mut self) {
        return;
    }

    fn get_exploration_probs(&self, action_space: &ActionSpace) -> Vec<f64> {
        return vec![self.confidence_level/action_space.size as f64; action_space.size];
    }

    fn get_exploration_rate(&self) -> f64 {
        return self.confidence_level;
    }

    fn reset(&mut self) {
        self.action_counter = RefCell::new(FxHashMap::default());
        self.t = RefCell::new(1);
    }
}