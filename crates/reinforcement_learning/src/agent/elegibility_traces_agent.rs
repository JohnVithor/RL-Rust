use std::collections::HashMap;

use ndarray::{Array, Array1};

use crate::{action_selection::ActionSelection, agent::DiscreteAgent};

use super::GetNextQValue;

pub struct ElegibilityTracesAgent {
    action_selection: Box<dyn ActionSelection>,
    next_value_function: GetNextQValue,
    learning_rate: f64,
    default_values: Array1<f64>,
    default_value: f64,
    discount_factor: f64,
    lambda_factor: f64,
    trace: HashMap<usize, Array1<f64>>,
    policy: Array1<Array1<f64>>,
}

impl ElegibilityTracesAgent {
    pub fn new(
        action_selection: Box<dyn ActionSelection>,
        next_value_function: GetNextQValue,
        learning_rate: f64,
        default_value: f64,
        discount_factor: f64,
        lambda_factor: f64,
    ) -> Self {
        Self {
            action_selection,
            next_value_function,
            learning_rate,
            default_values: Array1::default(0),
            default_value,
            discount_factor,
            lambda_factor,
            trace: HashMap::default(),
            policy: Array1::default(0),
        }
    }
}

impl DiscreteAgent for ElegibilityTracesAgent {
    fn prepare(&mut self, n_obs: usize, n_actions: usize) {
        let v = Array::from_elem((n_actions,), self.default_value);
        self.policy = Array1::from_elem((n_obs,), v.clone());
        self.default_values = v;
    }

    fn update(
        &mut self,
        curr_obs: usize,
        curr_action: usize,
        reward: f64,
        terminated: bool,
        next_obs: usize,
        next_action: usize,
    ) -> f64 {
        let next_q_values: &Array1<f64> = self.policy.get(next_obs).unwrap_or(&self.default_values);

        let future_q_value: f64 = (self.next_value_function)(
            next_q_values,
            next_action,
            &self
                .action_selection
                .get_exploration_probs(next_obs, next_q_values),
        );

        let curr_q_values: &Array1<f64> = self.policy.get(curr_obs).unwrap_or(&self.default_values);
        let temporal_difference: f64 =
            reward + self.discount_factor * future_q_value - curr_q_values[curr_action];

        let curr_trace = self
            .trace
            .entry(curr_obs)
            .or_insert(self.default_values.clone());
        curr_trace[curr_action] += 1.0;

        for (obs, trace_values) in &mut self.trace {
            for (action, value) in trace_values.iter_mut().enumerate() {
                self.policy.get_mut(*obs).unwrap()[action] =
                    self.learning_rate * temporal_difference * *value;
                *value *= self.discount_factor * self.lambda_factor
            }
        }

        // self.trace.iter_mut().for_each(|v| {
        //     v.iter_mut().enumerate().for_each(|(i, v)| {
        //         self.policy.get_mut(i).unwrap()[i] = self.learning_rate * temporal_difference * *v;
        //         *v *= self.discount_factor * self.lambda_factor;
        //     })
        // });

        if terminated {
            self.action_selection.update();
            self.trace = HashMap::default();
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
