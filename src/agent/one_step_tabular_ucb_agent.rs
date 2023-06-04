use super::{Agent, GetNextQValue};
use crate::utils::argmax;
use fxhash::FxHashMap;
use std::fmt::Debug;
use std::hash::Hash;

pub struct OneStepTabularUCBAgent<T: Hash + PartialEq + Eq + Clone + Debug, const COUNT: usize> {
    // policy
    default: [f64; COUNT],
    policy: FxHashMap<T, [f64; COUNT]>,
    // policy update
    learning_rate: f64,
    discount_factor: f64,
    // action selection
    confidence_level: f64,
    action_counter: FxHashMap<T, [u128; COUNT]>,
    t: u128,
    training_error: Vec<f64>,
    get_next_q_value: GetNextQValue<COUNT>,
}

impl<T: Hash + PartialEq + Eq + Clone + Debug, const COUNT: usize>
    OneStepTabularUCBAgent<T, COUNT>
{
    pub fn new(
        // policy
        default_value: f64,
        // policy update
        learning_rate: f64,
        discount_factor: f64,
        // action selection
        confidence_level: f64,
        get_next_q_value: GetNextQValue<COUNT>,
    ) -> Self {
        return Self {
            default: [default_value; COUNT],
            policy: FxHashMap::default(),
            learning_rate,
            discount_factor,
            confidence_level,
            action_counter: FxHashMap::default(),
            t: 0,
            training_error: vec![],
            get_next_q_value,
        };
    }
}

impl<T: Hash + PartialEq + Eq + Clone + Debug, const COUNT: usize> Agent<T, COUNT>
    for OneStepTabularUCBAgent<T, COUNT>
{
    fn reset(&mut self) {
        self.policy = FxHashMap::default();
        self.training_error = vec![];
    }

    fn get_action(&mut self, obs: &T) -> usize {
        let values: &[f64; COUNT] = self.policy.get(&obs).unwrap_or(&self.default);
        let obs_actions: &mut [u128; COUNT] =
            self.action_counter.entry(obs.clone()).or_insert([0; COUNT]);
        let mut ucbs: [f64; COUNT] = [0.0; COUNT];
        for i in 0..COUNT {
            ucbs[i] = values[i]
                + self.confidence_level
                    * ((self.t as f64).ln() / (obs_actions[i] as f64 + f64::MIN_POSITIVE)).sqrt()
        }
        let action = argmax(&ucbs);
        obs_actions[action] += 1;
        self.t += 1;
        return action;
    }

    fn update(
        &mut self,
        curr_obs: &T,
        curr_action: usize,
        reward: f64,
        _terminated: bool,
        next_obs: &T,
        next_action: usize,
    ) {
        let next_q_values: &[f64; COUNT] = self.policy.get(next_obs).unwrap_or(&self.default);
        let future_q_value: f64 =
            (self.get_next_q_value)(next_q_values, next_action, self.confidence_level);
        let curr_q_values: &mut [f64; COUNT] = self
            .policy
            .entry(curr_obs.clone())
            .or_insert(self.default.clone());
        let temporal_difference: f64 =
            reward + self.discount_factor * future_q_value - curr_q_values[curr_action];
        curr_q_values[curr_action] =
            curr_q_values[curr_action] + self.learning_rate * temporal_difference;
        self.training_error.push(temporal_difference);
    }

    fn get_training_error(&self) -> &Vec<f64> {
        return &self.training_error;
    }
}
