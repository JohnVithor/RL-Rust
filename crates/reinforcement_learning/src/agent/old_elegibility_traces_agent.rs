use super::DiscreteAgent;
use crate::action_selection::ActionSelection;
use crate::policy::DiscretePolicy;

use environments::env::DiscreteAction;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::{Index, IndexMut};

pub struct ElegibilityTracesAgent<'a, T: Hash + PartialEq + Eq + Clone + Debug, A: DiscreteAction>
where
    [(); A::RANGE]: Sized,
{
    // policy update
    discount_factor: f32,
    lambda_factor: f32,
    trace: HashMap<T, [f32; A::RANGE]>,
    get_next_q_value: fn(&[f32; A::RANGE], A, &[f32; A::RANGE]) -> f32,
    pub policy: &'a mut dyn DiscretePolicy<T, A>,
    action_selection: &'a mut dyn ActionSelection<T, A>,
}

impl<'a, T: Hash + PartialEq + Eq + Clone + Debug, A: DiscreteAction>
    ElegibilityTracesAgent<'a, T, A>
where
    [(); A::RANGE]: Sized,
{
    pub fn new(
        // policy update
        discount_factor: f32,
        lambda_factor: f32,
        get_next_q_value: fn(&[f32; A::RANGE], A, &[f32; A::RANGE]) -> f32,
        policy: &'a mut dyn DiscretePolicy<T, A>,
        action_selection: &'a mut dyn ActionSelection<T, A>,
    ) -> Self {
        Self {
            policy,
            trace: HashMap::default(),
            discount_factor,
            lambda_factor,
            get_next_q_value,
            action_selection,
        }
    }
}

impl<'a, T: Hash + PartialEq + Eq + Clone + Debug, A: DiscreteAction + Copy + Debug>
    DiscreteAgent<'a, T, A> for ElegibilityTracesAgent<'a, T, A>
where
    [f32]: Index<A, Output = f32>,
    [f32]: IndexMut<A, Output = f32>,
    [(); A::RANGE]: Sized,
{
    fn get_policy(&self) -> &dyn DiscretePolicy<T, A> {
        self.policy
    }

    fn set_future_q_value_func(&mut self, func: fn(&[f32; A::RANGE], A, &[f32; A::RANGE]) -> f32) {
        self.get_next_q_value = func;
    }

    fn set_action_selector(&mut self, action_selecter: &'a mut dyn ActionSelection<T, A>) {
        self.action_selection = action_selecter;
    }

    fn reset(&mut self) {
        self.action_selection.reset();
        self.policy.reset();
    }

    fn get_action(&mut self, obs: &T) -> A {
        self.action_selection
            .get_action(obs, &self.policy.predict(obs))
    }

    fn update(
        &mut self,
        curr_obs: &T,
        curr_action: A,
        reward: f32,
        terminated: bool,
        next_obs: &T,
        next_action: A,
    ) -> f32 {
        let next_q_values: [f32; A::RANGE] = self.policy.get_values(next_obs);
        let future_q_value: f32 = (self.get_next_q_value)(
            &next_q_values,
            next_action,
            &self
                .action_selection
                .get_exploration_probs(next_obs, &next_q_values),
        );
        let curr_q_values: [f32; A::RANGE] = self.policy.get_values(curr_obs);
        let temporal_difference: f32 =
            reward + self.discount_factor * future_q_value - curr_q_values[curr_action];

        let curr_trace: &mut [f32; A::RANGE] = self
            .trace
            .entry(curr_obs.clone())
            .or_insert([0.0; A::RANGE]);
        curr_trace[curr_action] += 1.0;

        for (obs, trace_values) in &mut self.trace {
            for (action, value) in trace_values.iter_mut().enumerate() {
                self.policy
                    .update(obs, action.into(), next_obs, temporal_difference * *value);
                *value *= self.discount_factor * self.lambda_factor
            }
        }

        self.policy.after_update();
        if terminated {
            self.trace = HashMap::default();
            self.action_selection.update();
        }
        temporal_difference
    }
}
