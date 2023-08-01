use super::{Agent, GetNextQValue};
use crate::action_selection::ActionSelection;
use crate::policy::Policy;
use std::fmt::Debug;

pub struct OneStepAgent<'a, T: Clone + Debug, const COUNT: usize> {
    // policy update
    discount_factor: f64,
    get_next_q_value: GetNextQValue<COUNT>,
    policy: &'a mut dyn Policy<T, COUNT>,
    action_selection: &'a mut dyn ActionSelection<T, COUNT>,
}

impl<'a, T: Clone + Debug, const COUNT: usize> OneStepAgent<'a, T, COUNT> {
    pub fn new(
        discount_factor: f64,
        get_next_q_value: GetNextQValue<COUNT>,
        policy: &'a mut dyn Policy<T, COUNT>,
        action_selection: &'a mut dyn ActionSelection<T, COUNT>,
    ) -> Self {
        Self {
            policy,
            discount_factor,
            action_selection,
            get_next_q_value,
        }
    }
}

impl<'a, T: Clone + Debug, const COUNT: usize> Agent<'a, T, COUNT> for OneStepAgent<'a, T, COUNT> {
    fn set_future_q_value_func(&mut self, func: GetNextQValue<COUNT>) {
        self.get_next_q_value = func;
    }

    fn set_action_selector(&mut self, action_selecter: &'a mut dyn ActionSelection<T, COUNT>) {
        self.action_selection = action_selecter;
    }

    fn reset(&mut self) {
        self.action_selection.reset();
        self.policy.reset();
    }

    fn get_action(&mut self, obs: &T) -> usize {
        self.action_selection
            .get_action(obs, &self.policy.predict(obs))
    }

    fn update(
        &mut self,
        curr_obs: &T,
        curr_action: usize,
        reward: f64,
        terminated: bool,
        next_obs: &T,
        next_action: usize,
    ) -> f64 {
        let next_q_values: [f64; COUNT] = self.policy.get_values(next_obs);
        let probs = &self
            .action_selection
            .get_exploration_probs(next_obs, &next_q_values);
        let future_q_value: f64 = (self.get_next_q_value)(&next_q_values, next_action, probs);
        let curr_q_values: [f64; COUNT] = self.policy.get_values(curr_obs);
        let temporal_difference: f64 =
            reward + self.discount_factor * future_q_value - curr_q_values[curr_action];

        let _error = self
            .policy
            .update(curr_obs, curr_action, next_obs, temporal_difference);

        self.policy.after_update();
        if terminated {
            self.action_selection.update();
        }
        temporal_difference
    }
}
