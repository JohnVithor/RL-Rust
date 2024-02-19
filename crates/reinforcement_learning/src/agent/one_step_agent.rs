use ndarray::{Array, Array1};

use crate::{action_selection::DiscreteObsDiscreteActionSelection, agent::FullDiscreteAgent};

use super::GetNextQValue;

pub struct OneStepAgent {
    action_selection: Box<dyn DiscreteObsDiscreteActionSelection>,
    next_value_function: GetNextQValue,
    learning_rate: f32,
    default_values: Array1<f32>,
    default_value: f32,
    discount_factor: f32,
    policy: Array1<Array1<f32>>,
}

impl OneStepAgent {
    pub fn new(
        action_selection: Box<dyn DiscreteObsDiscreteActionSelection>,
        next_value_function: GetNextQValue,
        learning_rate: f32,
        default_value: f32,
        discount_factor: f32,
    ) -> Self {
        Self {
            action_selection,
            next_value_function,
            learning_rate,
            default_values: Array1::default(0),
            default_value,
            discount_factor,
            policy: Array1::default(0),
        }
    }
}

impl FullDiscreteAgent for OneStepAgent {
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

        let value = self.policy.get(curr_obs).unwrap_or(&self.default_values)[curr_action];
        self.policy.get_mut(curr_obs).unwrap()[curr_action] =
            value + self.learning_rate * temporal_difference;

        if terminated {
            self.action_selection.update();
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
