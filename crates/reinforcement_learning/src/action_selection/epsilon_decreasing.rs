use super::{ContinuousObsDiscreteActionSelection, DiscreteObsDiscreteActionSelection};
use ndarray::{Array, Array1};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::rc::Rc;
use utils::argmax;

#[derive(Clone)]
pub struct EpsilonDecreasing {
    initial_epsilon: f32,
    epsilon: f32,
    epsilon_decay: Rc<dyn Fn(f32) -> f32>,
    final_epsilon: f32,
    rng: SmallRng,
}

impl Default for EpsilonDecreasing {
    fn default() -> Self {
        Self::new(1.0, Rc::new(|x| x - 0.01), 0.0, 42)
    }
}

impl EpsilonDecreasing {
    pub fn new(
        epsilon: f32,
        epsilon_decay: Rc<dyn Fn(f32) -> f32>,
        final_epsilon: f32,
        seed: u64,
    ) -> Self {
        Self {
            initial_epsilon: epsilon,
            epsilon,
            epsilon_decay,
            final_epsilon,
            rng: SmallRng::seed_from_u64(seed),
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

    fn update(&mut self) {
        let new_epsilon: f32 = (self.epsilon_decay)(self.epsilon);
        self.epsilon = if self.final_epsilon > new_epsilon {
            self.epsilon
        } else {
            new_epsilon
        };
    }

    fn get_exploration_probs(&mut self, values: &Array1<f32>) -> Array1<f32> {
        let mut policy_probs: Array1<f32> =
            Array::from_elem((values.len(),), self.epsilon / values.len() as f32);
        policy_probs[argmax(values.iter())] = 1.0 - self.epsilon;
        policy_probs
    }

    fn reset(&mut self) {
        self.epsilon = self.initial_epsilon;
    }
}

impl DiscreteObsDiscreteActionSelection for EpsilonDecreasing {
    fn get_action(&mut self, _obs: usize, values: &Array1<f32>) -> usize {
        self.get_action(values)
    }

    fn update(&mut self, _reward: f32) {
        self.update();
    }

    fn get_exploration_probs(&mut self, _obs: usize, values: &Array1<f32>) -> Array1<f32> {
        self.get_exploration_probs(values)
    }

    fn reset(&mut self) {
        self.reset()
    }
}

impl ContinuousObsDiscreteActionSelection for EpsilonDecreasing {
    fn get_action(&mut self, values: &Array1<f32>) -> usize {
        self.get_action(values)
    }

    fn update(&mut self, _reward: f32) {
        self.update();
    }

    fn get_exploration_probs(&mut self, values: &Array1<f32>) -> Array1<f32> {
        self.get_exploration_probs(values)
    }

    fn reset(&mut self) {
        self.reset()
    }
}
