use ndarray::{Array, Array1};

use crate::{action_selection::DiscreteObsDiscreteActionSelection, agent::DDAgent};

use super::GetNextQValue;

pub struct ElegibilityTracesAgent {
    action_selection: Box<dyn DiscreteObsDiscreteActionSelection>,
    next_value_function: GetNextQValue,
    learning_rate: f32,
    default_values: Array1<f32>,
    default_value: f32,
    discount_factor: f32,
    lambda_factor: f32,
    trace: Vec<(usize, Array1<f32>)>,
    policy: Array1<Array1<f32>>,
    reward_sum: f32,
    step_counter: u32,
}

impl ElegibilityTracesAgent {
    pub fn new(
        action_selection: Box<dyn DiscreteObsDiscreteActionSelection>,
        next_value_function: GetNextQValue,
        learning_rate: f32,
        default_value: f32,
        discount_factor: f32,
        lambda_factor: f32,
    ) -> Self {
        Self {
            action_selection,
            next_value_function,
            learning_rate,
            default_values: Array1::default(0),
            default_value,
            discount_factor,
            lambda_factor,
            trace: Vec::default(),
            policy: Array1::default(0),
            reward_sum: 0.0,
            step_counter: 0,
        }
    }
}

impl DDAgent for ElegibilityTracesAgent {
    fn prepare(&mut self, n_obs: usize, n_actions: usize) {
        let v = Array::from_elem((n_actions,), self.default_value);
        self.policy = Array1::from_elem((n_obs,), v.clone());
        self.default_values = v;
    }

    fn update(
        &mut self,
        curr_obs: usize,
        curr_action: usize,
        reward: f32,
        terminated: bool,
        next_obs: usize,
        next_action: usize,
    ) -> f32 {
        self.reward_sum += reward;
        self.step_counter += 1;
        let next_q_values: &Array1<f32> = self.policy.get(next_obs).unwrap_or(&self.default_values);

        let future_q_value: f32 = (self.next_value_function)(
            next_q_values,
            next_action,
            &self
                .action_selection
                .get_exploration_probs(next_obs, next_q_values),
        );

        let curr_q_values: &Array1<f32> = self.policy.get(curr_obs).unwrap_or(&self.default_values);
        let temporal_difference: f32 = reward
            + if terminated {
                0.0
            } else {
                self.discount_factor * future_q_value
            }
            - curr_q_values[curr_action];

        let mut found = false;
        for (obs, trace_values) in &mut self.trace {
            if *obs == curr_obs {
                trace_values[curr_action] += 1.0;
                found = true;
            }
            for (action, value) in trace_values.iter_mut().enumerate() {
                let policy_values = self.policy.get_mut(*obs).unwrap();
                policy_values[action] += self.learning_rate * temporal_difference * *value;
                *value *= self.discount_factor * self.lambda_factor
            }
        }
        if !found {
            let mut trace_values = self.default_values.clone();
            trace_values[curr_action] += 1.0;
            for (action, value) in trace_values.iter_mut().enumerate() {
                let policy_values = self.policy.get_mut(curr_obs).unwrap();
                policy_values[action] += self.learning_rate * temporal_difference * *value;
                *value *= self.discount_factor * self.lambda_factor
            }
            self.trace.push((curr_obs, trace_values));
        }

        if terminated {
            self.action_selection
                .update(self.reward_sum / self.step_counter as f32);
            self.reward_sum = 0.0;
            self.step_counter = 0;
            self.trace.clear();
        }
        temporal_difference
    }

    fn get_action(&mut self, obs: usize) -> usize {
        self.action_selection
            .get_action(obs, self.policy.get(obs).unwrap_or(&self.default_values))
    }

    fn reset(&mut self) {
        self.policy.fill(self.default_values.clone());
        self.action_selection.reset();
    }
}
