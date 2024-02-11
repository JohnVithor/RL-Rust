use super::ActionSelection;
use ndarray::{Array, Array1};
use rand::Rng;
use std::rc::Rc;
use utils::argmax;

#[derive(Clone)]
pub struct UniformEpsilonGreed {
    pub initial_epsilon: f64,
    pub epsilon: f64,
    epsilon_decay: Rc<dyn Fn(f64) -> f64>,
    final_epsilon: f64,
}

impl Default for UniformEpsilonGreed {
    fn default() -> Self {
        Self::new(1.0, Rc::new(|x| x - 0.01), 0.0)
    }
}

impl UniformEpsilonGreed {
    pub fn new(epsilon: f64, epsilon_decay: Rc<dyn Fn(f64) -> f64>, final_epsilon: f64) -> Self {
        Self {
            initial_epsilon: epsilon,
            epsilon,
            epsilon_decay,
            final_epsilon,
        }
    }

    fn should_explore(&self) -> bool {
        self.epsilon != 0.0 && rand::thread_rng().gen_range(0.0..1.0) < self.epsilon
    }
}

impl ActionSelection for UniformEpsilonGreed {
    fn get_action(&mut self, _obs: usize, values: &Array1<f64>) -> usize {
        if self.should_explore() {
            rand::thread_rng().gen_range(0..values.len())
        } else {
            argmax(values.iter())
        }
    }

    fn update(&mut self) {
        let new_epsilon: f64 = (self.epsilon_decay)(self.epsilon);
        self.epsilon = if self.final_epsilon > new_epsilon {
            self.epsilon
        } else {
            new_epsilon
        };
    }

    fn get_exploration_probs(&mut self, _obs: usize, values: &Array1<f64>) -> Array1<f64> {
        let mut policy_probs: Array1<f64> =
            Array::from_elem((values.len(),), self.epsilon / values.len() as f64);
        policy_probs[argmax(values.iter())] = 1.0 - self.epsilon;
        policy_probs
    }

    fn reset(&mut self) {
        self.epsilon = self.initial_epsilon;
    }
}
