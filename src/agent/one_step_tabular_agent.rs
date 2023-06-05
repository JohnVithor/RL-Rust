use super::{Agent, GetNextQValue};
use crate::action_selection::{ActionSelection, EnumActionSelection};
use fxhash::FxHashMap;
use std::fmt::Debug;
use std::hash::Hash;

pub struct OneStepTabularAgent<T: Hash + PartialEq + Eq + Clone + Debug, const COUNT: usize> {
    // policy
    default: [f64; COUNT],
    policy: FxHashMap<T, [f64; COUNT]>,
    // policy update
    learning_rate: f64,
    discount_factor: f64,
    training_error: Vec<f64>,
    action_selection: EnumActionSelection<T, COUNT>,
    get_next_q_value: GetNextQValue<COUNT>,
}

impl<T: Hash + PartialEq + Eq + Clone + Debug, const COUNT: usize>
    OneStepTabularAgent<T, COUNT>
{
    pub fn new(
        // policy
        default_value: f64,
        // policy update
        learning_rate: f64,
        discount_factor: f64,
        action_selection: EnumActionSelection<T, COUNT>,
        get_next_q_value: GetNextQValue<COUNT>,
    ) -> Self {
        return Self {
            default: [default_value; COUNT],
            policy: FxHashMap::default(),
            learning_rate,
            discount_factor,
            action_selection,
            training_error: vec![],
            get_next_q_value,
        };
    }
}

impl<T: Hash + PartialEq + Eq + Clone + Debug, const COUNT: usize> Agent<T, COUNT>
    for OneStepTabularAgent<T, COUNT>
{
    fn set_future_q_value_func(&mut self, func: GetNextQValue<COUNT>) {
        self.get_next_q_value = func;
    }

    fn set_action_selector(&mut self, action_selecter: EnumActionSelection<T, COUNT>) {
        self.action_selection = action_selecter;
    }

    fn reset(&mut self) {
        self.action_selection.reset();
        self.policy = FxHashMap::default();
    }

    fn get_action(&mut self, obs: &T) -> usize {
        let values: &mut [f64; COUNT] = self
            .policy
            .entry(obs.clone())
            .or_insert(self.default.clone());
        return self.action_selection.get_action(obs, values);
    }

    fn update(
        &mut self,
        curr_obs: &T,
        curr_action: usize,
        reward: f64,
        terminated: bool,
        next_obs: &T,
        next_action: usize,
    ) {
        let next_q_values: &[f64; COUNT] = self.policy.get(next_obs).unwrap_or(&self.default);
        let future_q_value: f64 = (self.get_next_q_value)(
            next_q_values,
            next_action,
            &self
                .action_selection
                .get_exploration_probs(next_obs, next_q_values),
        );
        let curr_q_values: &mut [f64; COUNT] = self
            .policy
            .entry(curr_obs.clone())
            .or_insert(self.default.clone());
        let temporal_difference: f64 =
            reward + self.discount_factor * future_q_value - curr_q_values[curr_action];
        curr_q_values[curr_action] =
            curr_q_values[curr_action] + self.learning_rate * temporal_difference;
        self.training_error.push(temporal_difference);
        if terminated {
            self.action_selection.update();
        }
    }

    fn get_training_error(&self) -> &Vec<f64> {
        return &self.training_error;
    }
}
