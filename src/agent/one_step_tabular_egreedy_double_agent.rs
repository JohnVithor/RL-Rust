use super::{Agent, GetNextQValue};
use crate::utils::argmax;
use fxhash::FxHashMap;
use rand::{distributions::Uniform, prelude::Distribution};
use std::fmt::Debug;
use std::hash::Hash;

pub struct OneStepTabularEGreedyDoubleAgent<
    T: Hash + PartialEq + Eq + Clone + Debug,
    const COUNT: usize,
> {
    // policy
    default: [f64; COUNT],
    alpha_policy: FxHashMap<T, [f64; COUNT]>,
    beta_policy: FxHashMap<T, [f64; COUNT]>,
    policy_flag: bool,
    // policy update
    learning_rate: f64,
    discount_factor: f64,
    // action selection
    exploration_decider: Uniform<f64>,
    rand_action_selecter: Uniform<usize>,
    initial_epsilon: f64,
    epsilon: f64,
    epsilon_decay: f64,
    final_epsilon: f64,
    training_error: Vec<f64>,
    get_next_q_value: GetNextQValue<COUNT>,
}

impl<T: Hash + PartialEq + Eq + Clone + Debug, const COUNT: usize>
    OneStepTabularEGreedyDoubleAgent<T, COUNT>
{
    pub fn new(
        // policy
        default_value: f64,
        // policy update
        learning_rate: f64,
        discount_factor: f64,
        // action selection
        initial_epsilon: f64,
        epsilon_decay: f64,
        final_epsilon: f64,
        get_next_q_value: GetNextQValue<COUNT>,
    ) -> Self {
        return Self {
            default: [default_value; COUNT],
            alpha_policy: FxHashMap::default(),
            beta_policy: FxHashMap::default(),
            policy_flag: true,
            learning_rate,
            discount_factor,
            exploration_decider: Uniform::from(0.0..1.0),
            rand_action_selecter: Uniform::from(0..COUNT),
            initial_epsilon,
            epsilon: initial_epsilon,
            epsilon_decay,
            final_epsilon,
            training_error: vec![],
            get_next_q_value,
        };
    }

    fn decay_epsilon(&mut self) {
        let new_epsilon: f64 = self.epsilon - self.epsilon_decay;
        self.epsilon = if self.final_epsilon > new_epsilon {
            self.epsilon
        } else {
            new_epsilon
        };
    }

    fn should_explore(&self) -> bool {
        return self.exploration_decider.sample(&mut rand::thread_rng()) < self.epsilon;
    }
}

impl<T: Hash + PartialEq + Eq + Clone + Debug, const COUNT: usize> Agent<T, COUNT>
    for OneStepTabularEGreedyDoubleAgent<T, COUNT>
{
    fn reset(&mut self) {
        self.epsilon = self.initial_epsilon;
        self.alpha_policy = FxHashMap::default();
        self.beta_policy = FxHashMap::default();
    }

    fn get_action(&mut self, obs: &T) -> usize {
        if self.should_explore() {
            return self.rand_action_selecter.sample(&mut rand::thread_rng());
        } else {
            let a_values = self.alpha_policy.get(&obs);
            let b_values = self.beta_policy.get(&obs);
            if a_values.is_none() && b_values.is_none() {
                return self.rand_action_selecter.sample(&mut rand::thread_rng());
            }
            let mut temp;
            if a_values.is_some() {
                temp = a_values.unwrap().clone();
                for (i, v) in b_values.unwrap_or(&self.default).iter().enumerate() {
                    temp[i] += v;
                    temp[i] /= 2.0;
                }
            } else {
                temp = b_values.unwrap().clone();
                for (i, v) in b_values.unwrap_or(&self.default).iter().enumerate() {
                    temp[i] += v;
                    temp[i] /= 2.0;
                }
            }

            return argmax(&temp);
        }
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
        let future_q_value: f64 = (self.get_next_q_value)(next_q_values, next_action, self.epsilon);
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
            self.decay_epsilon();
        }
        self.policy_flag = !self.policy_flag;
    }

    fn get_training_error(&self) -> &Vec<f64> {
        return &self.training_error;
    }
}
