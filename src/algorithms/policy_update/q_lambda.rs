use std::cell::{RefCell, RefMut};

use super::PolicyUpdate;

use crate::{env::{Observation, ActionSpace}, Policy, utils::argmax};

pub struct QLambda {
    learning_rate: f64,
    discount_factor: f64,
    pub trace: RefCell<Policy>,
    lambda_factor: f64
}

impl QLambda {
    pub fn new(
        learning_rate: f64,
        discount_factor: f64,
        lambda_factor: f64,
        default_value: f64,
        action_space: ActionSpace
    ) -> Self {
        return Self{
            learning_rate,
            discount_factor,
            trace: RefCell::new(Policy::new(default_value, action_space)),
            lambda_factor
        }
    }
}

impl PolicyUpdate for QLambda {
    fn update(
        &mut self,
        curr_obs: Observation,
        curr_action: usize,
        next_obs: Observation,
        next_action: usize,
        reward: f64,
        terminated: bool,
        policy: &mut Policy
    ) {
        let next_q_values: &Vec<f64> = policy.get_ref(next_obs);
        let best_next_action: usize = argmax(&next_q_values);
        // O valor dentro do if é a diferença entre o SARSA e o QLearning
        // Aqui o melhor valor é utilizado (valor da melhor ação)
        let future_q_value: f64 = if !terminated {
            next_q_values[best_next_action]
        } else {
            0.0
        };
        let values: &mut Vec<f64> = policy.get_mut(curr_obs.clone());
        let temporal_difference: f64 = reward + self.discount_factor * future_q_value - values[curr_action];
        let mut trace: RefMut<Policy> = self.trace.borrow_mut();
        trace.get_mut(curr_obs.clone())[curr_action] += 1.0;

        for (obs, values) in &mut policy.values {
            let t_values: &mut Vec<f64> = trace.get_mut(obs.clone());
            for i in 0..values.len() {
                values[i] = values[i] + self.learning_rate * temporal_difference * t_values[i];
                if next_action == best_next_action {
                    t_values[i] = self.discount_factor * self.lambda_factor * t_values[i]
                } else {
                    t_values[i] = 0.0;
                }
            }
        }
    }
}