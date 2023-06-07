use super::{Agent, GetNextQValue};
use crate::action_selection::{ActionSelection, EnumActionSelection};
use crate::policy::{EnumPolicy, Policy};
use fxhash::FxHashMap;
use std::fmt::Debug;
use std::hash::Hash;

pub struct ElegibilityTracesTabularAgent<
    T: Hash + PartialEq + Eq + Clone + Debug,
    const COUNT: usize,
> {
    policy: EnumPolicy<T, COUNT>,
    // policy update
    learning_rate: f64,
    discount_factor: f64,
    action_selection: EnumActionSelection<T, COUNT>,
    lambda_factor: f64,
    trace: FxHashMap<T, [f64; COUNT]>,
    get_next_q_value: GetNextQValue<COUNT>,
}

impl<T: Hash + PartialEq + Eq + Clone + Debug, const COUNT: usize>
    ElegibilityTracesTabularAgent<T, COUNT>
{
    pub fn new(
        policy: EnumPolicy<T, COUNT>,
        // policy update
        learning_rate: f64,
        discount_factor: f64,
        action_selection: EnumActionSelection<T, COUNT>,
        lambda_factor: f64,
        get_next_q_value: GetNextQValue<COUNT>,
    ) -> Self {
        Self {
            policy,
            trace: FxHashMap::default(),
            learning_rate,
            discount_factor,
            action_selection,
            lambda_factor,
            get_next_q_value,
        }
    }
}

impl<T: Hash + PartialEq + Eq + Clone + Debug, const COUNT: usize> Agent<T, COUNT>
    for ElegibilityTracesTabularAgent<T, COUNT>
{
    fn set_future_q_value_func(&mut self, func: GetNextQValue<COUNT>) {
        self.get_next_q_value = func;
    }

    fn set_action_selector(&mut self, action_selecter: EnumActionSelection<T, COUNT>) {
        self.action_selection = action_selecter;
    }

    fn reset(&mut self) {
        self.action_selection.reset();
        self.policy.reset();
    }

    fn get_action(&mut self, obs: &T) -> usize {
        self.action_selection.get_action(obs, &self.policy.predict(obs))
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
        let next_q_values: &[f64; COUNT] = self.policy.get_values(next_obs);
        let future_q_value: f64 = (self.get_next_q_value)(
            next_q_values,
            next_action,
            &self
                .action_selection
                .get_exploration_probs(next_obs, next_q_values),
        );
        let curr_q_values: &[f64; COUNT] = self.policy.get_values(curr_obs);
        let temporal_difference: f64 =
            reward + self.discount_factor * future_q_value - curr_q_values[curr_action];

        let curr_trace: &mut [f64; COUNT] =
            self.trace.entry(curr_obs.clone()).or_insert([0.0; COUNT]);
        curr_trace[curr_action] += 1.0;

        for (obs, trace_values) in &mut self.trace {
            for (action, value) in trace_values.iter_mut().enumerate() {
                self.policy.update(
                    obs,
                    action,
                    self.learning_rate * temporal_difference * *value,
                );
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
