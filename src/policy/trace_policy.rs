use std::hash::Hash;
use fxhash::FxHashMap;

use crate::{env::ActionSpace, utils::argmax};

use super::Policy;

pub struct TracePolicy<T> {
    policy: Box<dyn Policy<T>>,
    pub trace: FxHashMap<T, ndarray::Array1<f64>>,
    lambda_factor: f64,
    mode: TraceMode
}

#[derive(Debug, Clone)]
pub enum TraceMode {
    SARSA,
    QLearning
}


impl<T: Hash+PartialEq+Eq+Clone> TracePolicy<T> {
    pub fn new(policy: Box<dyn Policy<T>>, lambda_factor: f64, mode: TraceMode) -> Self {
        return Self {
            policy,
            trace: FxHashMap::default(),
            lambda_factor,
            mode
        };
    }
}

impl<T: Hash+PartialEq+Eq+Clone> Policy<T> for TracePolicy<T>{
    fn get_values(&mut self, curr_obs: T) -> &ndarray::Array1<f64> {
        return self.policy.get_values(curr_obs)
    }

    fn predict(&mut self, curr_obs: T) -> &ndarray::Array1<f64> {
        return self.policy.predict(curr_obs);
    }

    fn update_values(&mut self, curr_obs: T, curr_action: usize, next_obs: T, next_action: usize, temporal_difference: f64) {
        let best_next_action = argmax(self.policy.get_values(next_obs.clone()));
        let curr_trace = self.trace.entry(curr_obs).or_insert(self.policy.get_default().clone());
        curr_trace[curr_action] += 1.0;
        for (obs, e_values) in &mut self.trace {
            for i in 0..e_values.len() {
                self.policy.update_values(obs.clone(), curr_action, next_obs.clone(), next_action, temporal_difference * e_values[i]);
                match self.mode {
                    TraceMode::SARSA => {
                        e_values[i] = self.policy.get_discount_factor() * self.lambda_factor * e_values[i]
                    },
                    TraceMode::QLearning => {
                        if next_action == best_next_action {
                            e_values[i] = self.policy.get_discount_factor() * self.lambda_factor * e_values[i]
                        } else {
                            e_values[i] = 0.0;
                        }
                    },
                } 
            }
        }
    }

    fn get_action_space(&self) -> &ActionSpace {
        return &self.policy.get_action_space();
    }

    fn reset(&mut self) {
        self.policy.reset();
        self.trace = FxHashMap::default();
    }

    fn after_update(&mut self) {
        self.policy.after_update();
        
    }

    fn get_learning_rate(&self) -> f64 {
        return self.policy.get_learning_rate();
    }

    fn get_discount_factor(&self) -> f64 {
        return self.policy.get_discount_factor();
    }

    fn get_default(&self) -> &ndarray::Array1<f64> {
        return self.policy.get_default();
    }

    fn get_td(&mut self, curr_obs: T, curr_action: usize, reward: f64, future_q_value: f64) -> f64 {
        return self.policy.get_td(curr_obs, curr_action, reward, future_q_value);
    }
}
