use super::Agent;
use crate::action_selection::ActionSelection;
use crate::policy::DiscretePolicy;
use environments::env::DiscreteAction;
use std::fmt::Debug;
use std::ops::Index;

pub struct OneStepAgent<'a, T: Clone + Debug, A: DiscreteAction>
where
    [(); A::RANGE]: Sized,
{
    // policy update
    discount_factor: f64,
    get_next_q_value: fn(&[f64; A::RANGE], A, &[f64; A::RANGE]) -> f64,
    policy: &'a mut dyn DiscretePolicy<T, A>,
    action_selection: &'a mut dyn ActionSelection<T, A>,
}

impl<'a, T: Clone + Debug, A: DiscreteAction> OneStepAgent<'a, T, A>
where
    [(); A::RANGE]: Sized,
{
    pub fn new(
        discount_factor: f64,
        get_next_q_value: fn(&[f64; A::RANGE], A, &[f64; A::RANGE]) -> f64,
        policy: &'a mut dyn DiscretePolicy<T, A>,
        action_selection: &'a mut dyn ActionSelection<T, A>,
    ) -> Self {
        Self {
            policy,
            discount_factor,
            action_selection,
            get_next_q_value,
        }
    }
}

impl<'a, T: Clone + Debug, A: DiscreteAction + Debug + Copy> Agent<'a, T, A>
    for OneStepAgent<'a, T, A>
where
    [f64]: Index<A, Output = f64>,
    [(); A::RANGE]: Sized,
{
    fn set_future_q_value_func(&mut self, func: fn(&[f64; A::RANGE], A, &[f64; A::RANGE]) -> f64) {
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
        reward: f64,
        terminated: bool,
        next_obs: &T,
        next_action: A,
    ) -> f64 {
        let next_q_values: [f64; A::RANGE] = self.policy.get_values(next_obs);
        let probs = &self
            .action_selection
            .get_exploration_probs(next_obs, &next_q_values);
        let future_q_value: f64 = (self.get_next_q_value)(&next_q_values, next_action, probs);
        let curr_q_values: [f64; A::RANGE] = self.policy.get_values(curr_obs);
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
