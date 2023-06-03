use std::{hash::Hash, cell::{RefCell, RefMut}};

use fxhash::FxHashMap;

use crate::{env::ActionSpace, utils::argmax, policy::Policy};

use super::ActionSelection;

#[derive(Debug, Clone)]
pub struct UpperConfidenceBound<T> {
    action_counter: RefCell<FxHashMap<T, Vec<u128>>>,
    t: RefCell<u128>,
    confidence_level: f64
}

impl<T> UpperConfidenceBound<T> {
    pub fn new(confidence_level: f64) -> Self {
        return Self {
            action_counter: RefCell::new(FxHashMap::default()),
            t: RefCell::new(1),
            confidence_level
        }
    }
}

impl<T: Hash+PartialEq+Eq+Clone> ActionSelection<T> for UpperConfidenceBound<T> {
    fn get_action(&self, obs: &T,  policy: &mut RefMut<&mut dyn Policy<T>>) -> usize {
        let ac = policy.get_action_space().clone();
        let values = policy.predict(obs.clone());
        let mut action_counter = self.action_counter.borrow_mut();
        let obs_actions = action_counter.entry(obs.clone()).or_insert(vec![0;ac.size]);
        let mut t: RefMut<u128> = self.t.borrow_mut();
        let ucbs: ndarray::Array1<f64> = values.iter().enumerate().map(
            |(i, v)| { 
            return v + self.confidence_level * ((*t as f64).ln() / (obs_actions[i] as f64  + f64::MIN_POSITIVE)).sqrt()
        }).collect();
        let action = argmax(&ucbs);
        obs_actions[action] += 1;
        *t += 1;
        return action;
    }

    fn update(&mut self) {
        return;
    }

    fn get_exploration_probs(&self, action_space: &ActionSpace) -> ndarray::Array1<f64> {
        return ndarray::Array1::from_elem(action_space.size, self.confidence_level/action_space.size as f64);
    }

    fn get_exploration_rate(&self) -> f64 {
        return self.confidence_level;
    }

    fn reset(&mut self) {
        self.action_counter = RefCell::new(FxHashMap::default());
        self.t = RefCell::new(1);
    }
}