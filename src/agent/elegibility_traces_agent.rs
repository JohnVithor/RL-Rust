use super::{Agent, GetNextQValue};
use crate::action_selection::ActionSelection;
use crate::policy::Policy;
use fxhash::FxHashMap;
use std::fmt::Debug;
use std::hash::Hash;

pub struct ElegibilityTracesAgent<'a, T: Hash + PartialEq + Eq + Clone + Debug, const COUNT: usize>
{
    // policy update
    discount_factor: f64,
    lambda_factor: f64,
    trace: FxHashMap<T, [f64; COUNT]>,
    get_next_q_value: GetNextQValue<COUNT>,
    policy: &'a mut dyn Policy<T, COUNT>,
    action_selection: &'a mut dyn ActionSelection<T, COUNT>,
}

impl<'a, T: Hash + PartialEq + Eq + Clone + Debug, const COUNT: usize>
    ElegibilityTracesAgent<'a, T, COUNT>
{
    pub fn new(
        // policy update
        discount_factor: f64,
        lambda_factor: f64,
        get_next_q_value: GetNextQValue<COUNT>,
        policy: &'a mut dyn Policy<T, COUNT>,
        action_selection: &'a mut dyn ActionSelection<T, COUNT>,
    ) -> Self {
        Self {
            policy,
            trace: FxHashMap::default(),
            discount_factor,
            lambda_factor,
            get_next_q_value,
            action_selection,
        }
    }
}

impl<'a, T: Hash + PartialEq + Eq + Clone + Debug, const COUNT: usize> Agent<'a, T, COUNT>
    for ElegibilityTracesAgent<'a, T, COUNT>
{
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
        let future_q_value: f64 = (self.get_next_q_value)(
            &next_q_values,
            next_action,
            &self
                .action_selection
                .get_exploration_probs(next_obs, &next_q_values),
        );
        let curr_q_values: [f64; COUNT] = self.policy.get_values(curr_obs);
        let temporal_difference: f64 =
            reward + self.discount_factor * future_q_value - curr_q_values[curr_action];

        let curr_trace: &mut [f64; COUNT] =
            self.trace.entry(curr_obs.clone()).or_insert([0.0; COUNT]);
        curr_trace[curr_action] += 1.0;

        for (obs, trace_values) in &mut self.trace {
            for (action, value) in trace_values.iter_mut().enumerate() {
                self.policy
                    .update(obs, action, next_obs, temporal_difference * *value);
                *value *= self.discount_factor * self.lambda_factor
            }
        }

        self.policy.after_update();
        if terminated {
            self.trace = FxHashMap::default();
            self.action_selection.update();
        }
        temporal_difference
    }
}
