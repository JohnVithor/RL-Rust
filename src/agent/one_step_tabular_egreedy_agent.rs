use fxhash::FxHashMap;
use rand::{distributions::Uniform, prelude::Distribution};
use std::hash::Hash;
use crate::utils::argmax;
use std::fmt::Debug;
use super::{Agent, GetNextQValue};

pub struct OneStepTabularEGreedyAgent<T: Hash+PartialEq+Eq+Clone+Debug, const COUNT: usize> {
    // policy
    default: [f64; COUNT],
    policy: FxHashMap<T, [f64; COUNT]>,
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
    get_next_q_value: GetNextQValue<COUNT>
}

impl<T: Hash+PartialEq+Eq+Clone+Debug, const COUNT: usize> OneStepTabularEGreedyAgent<T, COUNT> {
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
            policy: FxHashMap::default(),
            learning_rate,
            discount_factor,
            exploration_decider: Uniform::from(0.0..1.0),
            rand_action_selecter: Uniform::from(0..COUNT),
            initial_epsilon,
            epsilon: initial_epsilon,
            epsilon_decay,
            final_epsilon,
            training_error: vec![],
            get_next_q_value
        }
    }   

    fn decay_epsilon(&mut self) {
        let new_epsilon: f64 = self.epsilon - self.epsilon_decay;
        self.epsilon = if self.final_epsilon > new_epsilon {self.epsilon} else {new_epsilon};
    }

    fn should_explore(&self) -> bool {
        return self.exploration_decider.sample(&mut rand::thread_rng()) < self.epsilon;
    }

}

impl<T: Hash+PartialEq+Eq+Clone+Debug, const COUNT: usize> Agent<T, COUNT> for OneStepTabularEGreedyAgent<T, COUNT> {
    fn reset(&mut self) {
        self.epsilon = self.initial_epsilon;
        self.policy = FxHashMap::default();
    }

    fn get_action(&self, obs: &T) -> usize {
        if self.should_explore() {
            return self.rand_action_selecter.sample(&mut rand::thread_rng());
        } else {
            match self.policy.get(obs) {
                Some(value) => argmax(value),
                None => self.rand_action_selecter.sample(&mut rand::thread_rng()),
            }
        }
    }

    fn update(
        &mut self,
        curr_obs: &T,
        curr_action: usize,
        reward: f64,
        terminated: bool,
        next_obs: &T,
        next_action: usize
    ) {
        let next_q_values: &[f64; COUNT] = self.policy.entry(next_obs.clone()).or_insert(self.default.clone());
        let future_q_value: f64 = (self.get_next_q_value)(next_q_values, next_action, self.epsilon);
        let curr_q_values: &mut [f64; COUNT] = self.policy.entry(curr_obs.clone()).or_insert(self.default.clone());
        let temporal_difference: f64 = reward + self.discount_factor * future_q_value - curr_q_values[curr_action];
        curr_q_values[curr_action] = curr_q_values[curr_action] + self.learning_rate * temporal_difference;
        self.training_error.push(temporal_difference);
        if terminated {
            self.decay_epsilon();
        }
    }

    fn get_training_error(&self) -> &Vec<f64>{
        return &self.training_error;
    }
}