use super::{ContinuousObsDiscreteActionSelection, DiscreteObsDiscreteActionSelection};
use ndarray::{Array, Array1};
use rand::{rngs::StdRng, Rng, SeedableRng};
use utils::argmax;

#[derive(Clone)]
pub struct EpsilonGreedy {
    epsilon: f32,
    rng: StdRng,
}

impl Default for EpsilonGreedy {
    fn default() -> Self {
        Self::new(0.1, 42)
    }
}

impl EpsilonGreedy {
    pub fn new(epsilon: f32, seed: u64) -> Self {
        Self {
            epsilon,
            rng: StdRng::seed_from_u64(seed),
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
}

impl DiscreteObsDiscreteActionSelection for EpsilonGreedy {
    fn get_action(&mut self, _obs: usize, values: &Array1<f32>) -> usize {
        self.get_action(values)
    }

    fn update(&mut self, _reward: f32) {}

    fn get_exploration_probs(&mut self, _obs: usize, values: &Array1<f32>) -> Array1<f32> {
        self.get_exploration_probs(values)
    }

    fn reset(&mut self) {}
}

impl ContinuousObsDiscreteActionSelection for EpsilonGreedy {
    fn get_action(&mut self, values: &Array1<f32>) -> usize {
        self.get_action(values)
    }

    fn update(&mut self, _reward: f32) {}

    fn get_exploration_probs(&mut self, values: &Array1<f32>) -> Array1<f32> {
        self.get_exploration_probs(values)
    }

    fn reset(&mut self) {}
}
