use super::{Agent, GetNextQValue};
use crate::action_selection::{ActionSelection, EnumActionSelection};
use fxhash::FxHashMap;
use std::fmt::Debug;
use std::hash::Hash;

pub struct OneStepTabularDoubleAgent<T: Hash + PartialEq + Eq + Clone + Debug, const COUNT: usize> {
    // policy
    default: [f64; COUNT],
    alpha_policy: FxHashMap<T, [f64; COUNT]>,
    beta_policy: FxHashMap<T, [f64; COUNT]>,
    policy_flag: bool,
    // policy update
    learning_rate: f64,
    discount_factor: f64,
    action_selection: EnumActionSelection<T, COUNT>,
    training_error: Vec<f64>,
    get_next_q_value: GetNextQValue<COUNT>,
}

impl<T: Hash + PartialEq + Eq + Clone + Debug, const COUNT: usize>
    OneStepTabularDoubleAgent<T, COUNT>
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
            alpha_policy: FxHashMap::default(),
            beta_policy: FxHashMap::default(),
            policy_flag: true,
            learning_rate,
            discount_factor,
            action_selection,
            training_error: vec![],
            get_next_q_value,
        };
    }
}

impl<T: Hash + PartialEq + Eq + Clone + Debug, const COUNT: usize> Agent<T, COUNT>
    for OneStepTabularDoubleAgent<T, COUNT>
{
    fn set_future_q_value_func(&mut self, func: GetNextQValue<COUNT>) {
        self.get_next_q_value = func;
    }

    fn set_action_selector(&mut self, action_selecter: EnumActionSelection<T, COUNT>) {
        self.action_selection = action_selecter;
    }
    
    fn reset(&mut self) {
        self.action_selection.reset();
        self.alpha_policy = FxHashMap::default();
        self.beta_policy = FxHashMap::default();
    }

    fn get_action(&mut self, obs: &T) -> usize {
        let mut values;
        let a_values = self.alpha_policy.get(&obs);
        let b_values = self.beta_policy.get(&obs);
        if a_values.is_none() && b_values.is_none() {
            return self.action_selection.get_action(obs, &self.default);
        }
        if a_values.is_some() {
            values = a_values.unwrap().clone();
            for (i, v) in b_values.unwrap_or(&self.default).iter().enumerate() {
                values[i] += v;
                values[i] /= 2.0;
            }
        } else {
            values = b_values.unwrap().clone();
            for (i, v) in b_values.unwrap_or(&self.default).iter().enumerate() {
                values[i] += v;
                values[i] /= 2.0;
            }
        }
        return self.action_selection.get_action(obs, &values);
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
        let query_policy = match self.policy_flag {
            true => &self.alpha_policy,
            false => &self.beta_policy,
        };
        let next_q_values: &[f64; COUNT] = query_policy.get(next_obs).unwrap_or(&self.default);
        let future_q_value: f64 = (self.get_next_q_value)(
            next_q_values,
            next_action,
            &self
                .action_selection
                .get_exploration_probs(next_obs, next_q_values),
        );
        let target_policy = match self.policy_flag {
            true => &mut self.beta_policy,
            false => &mut self.alpha_policy,
        };
        let curr_q_values: &mut [f64; COUNT] = target_policy
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
        self.policy_flag = !self.policy_flag;
    }

    fn get_training_error(&self) -> &Vec<f64> {
        return &self.training_error;
    }
}
