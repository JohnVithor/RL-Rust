use super::PolicyUpdate;

use crate::{env::Observation, Policy};

pub struct SarsaStep {
    learning_rate: f64,
    discount_factor: f64
}

impl SarsaStep {
    pub fn new(learning_rate: f64, discount_factor: f64) -> Self {
        return Self{learning_rate, discount_factor}
    }
}

impl PolicyUpdate for SarsaStep {
    fn update(
        &mut self,
        curr_obs:Observation,
        curr_action: usize,
        next_obs: Observation,
        next_action: usize,
        reward: f64,
        terminated: bool,
        policy: &mut Policy
    ) {
        let next_q_values: &Vec<f64> = policy.get_ref(next_obs.clone());
        // O valor dentro do else é a diferença entre o SARSA e o QLearning
        // Aqui o valor da próxima ação é utilizado
        let future_q_value = if !terminated {
            next_q_values[next_action]
        } else {
            0.0
        };
        let values: &mut Vec<f64> = policy.get_mut(curr_obs);
        let temporal_difference: f64 = reward + self.discount_factor * future_q_value - values[curr_action];
        values[curr_action] = values[curr_action] + self.learning_rate * temporal_difference;
    }
}

