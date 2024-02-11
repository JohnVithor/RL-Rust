use std::collections::HashMap;

use ndarray::Array1;
use utils::argmax;

use super::ActionSelection;

#[derive(Debug, Clone)]
pub struct UpperConfidenceBound {
    action_counter: HashMap<usize, Array1<u128>>,
    t: u128,
    confidence_level: f64,
}

impl UpperConfidenceBound {
    pub fn new(confidence_level: f64) -> Self {
        Self {
            action_counter: HashMap::default(),
            t: 1,
            confidence_level,
        }
    }
}

impl ActionSelection for UpperConfidenceBound {
    fn get_action(&mut self, obs: usize, values: &Array1<f64>) -> usize {
        let obs_actions = self
            .action_counter
            .entry(obs)
            .or_insert(Array1::from_elem(values.len(), 0));
        let mut ucbs = Array1::from_elem(values.len(), 0.0);
        for i in 0..values.len() {
            ucbs[i] = values[i]
                + self.confidence_level
                    * ((self.t as f64).ln() / (obs_actions[i] as f64 + f64::MIN_POSITIVE)).sqrt()
        }
        let action = argmax(ucbs.iter());
        obs_actions[action] += 1;
        self.t += 1;
        action
    }

    fn update(&mut self) {}

    fn get_exploration_probs(&mut self, obs: usize, values: &Array1<f64>) -> Array1<f64> {
        let obs_actions = self
            .action_counter
            .entry(obs)
            .or_insert(Array1::from_elem(values.len(), 0));
        let mut ucbs = Array1::from_elem(values.len(), 0.0);
        for i in 0..values.len() {
            ucbs[i] = values[i]
                + self.confidence_level
                    * ((self.t as f64).ln() / (obs_actions[i] as f64 + f64::MIN_POSITIVE)).sqrt()
        }
        let action = argmax(ucbs.iter());
        let mut probs = Array1::from_elem(values.len(), 0.0);
        probs[action] = 1.0;
        probs
    }

    fn reset(&mut self) {
        self.action_counter = HashMap::default();
        self.t = 1;
    }
}
