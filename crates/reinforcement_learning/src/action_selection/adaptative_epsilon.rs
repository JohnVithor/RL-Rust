use super::{ContinuousObsDiscreteActionSelection, DiscreteObsDiscreteActionSelection};
use ndarray::{Array, Array1};
use rand::{rngs::StdRng, Rng, SeedableRng};
use utils::argmax;

#[derive(Clone)]
pub struct AdaptativeEpsilon {
    rng: StdRng,
    min_reward: f32,
    max_reward: f32,
    min_epsilon: f32,
    max_epsilon: f32,
    pub epsilon: f32,
}

impl AdaptativeEpsilon {
    pub fn new(
        min_reward: f32,
        max_reward: f32,
        min_epsilon: f32,
        max_epsilon: f32,
        seed: u64,
    ) -> Self {
        Self {
            min_reward,
            max_reward,
            min_epsilon,
            max_epsilon,
            rng: StdRng::seed_from_u64(seed),
            epsilon: max_epsilon,
        }
    }

    fn should_explore(&mut self) -> bool {
        self.epsilon != 0.0 && self.rng.gen_range(0.0..1.0) <= self.epsilon
    }
    fn get_action(&mut self, values: &Array1<f32>) -> usize {
        if self.should_explore() {
            self.rng.gen_range(0..values.len())
        } else {
            argmax(values.iter())
        }
    }

    fn get_exploration_probs(&mut self, values: &Array1<f32>) -> Array1<f32> {
        let mut policy_probs: Array1<f32> =
            Array::from_elem((values.len(),), self.epsilon / values.len() as f32);
        policy_probs[argmax(values.iter())] = 1.0 - self.epsilon;
        policy_probs
    }

    fn reset(&mut self) {
        self.epsilon = self.max_epsilon;
    }

    fn update(&mut self, reward: f32) {
        if reward < self.min_reward {
            self.epsilon = self.max_epsilon;
        }
        let reward_range = self.max_reward - self.min_reward;
        let eps_range = self.max_epsilon - self.min_epsilon;
        let min_update = eps_range / reward_range;
        let new_eps = (self.max_reward - reward) * min_update;
        self.epsilon = if new_eps < self.min_epsilon {
            self.min_epsilon
        } else {
            new_eps
        };
    }
}

impl DiscreteObsDiscreteActionSelection for AdaptativeEpsilon {
    fn get_action(&mut self, _obs: usize, values: &Array1<f32>) -> usize {
        self.get_action(values)
    }

    fn update(&mut self, reward: f32) {
        self.update(reward);
    }

    fn get_exploration_probs(&mut self, _obs: usize, values: &Array1<f32>) -> Array1<f32> {
        self.get_exploration_probs(values)
    }

    fn reset(&mut self) {
        self.reset();
    }
}

impl ContinuousObsDiscreteActionSelection for AdaptativeEpsilon {
    fn get_action(&mut self, values: &Array1<f32>) -> usize {
        self.get_action(values)
    }

    fn update(&mut self, reward: f32) {
        self.update(reward);
    }

    fn get_exploration_probs(&mut self, values: &Array1<f32>) -> Array1<f32> {
        self.get_exploration_probs(values)
    }

    fn reset(&mut self) {
        self.reset();
    }
}
