use std::rc::Rc;

use super::{ContinuousObsDiscreteActionSelection, DiscreteObsDiscreteActionSelection};
use fastrand::Rng;
use ndarray::{Array, Array1};
use utils::argmax;

#[derive(Clone)]
pub struct AdaptativeEpsilon {
    min_epsilon: f32,
    max_epsilon: f32,
    min_reward: f32,
    max_reward: f32,
    eps_range: f32,
}

impl AdaptativeEpsilon {
    pub fn new(min_epsilon: f32, max_epsilon: f32, min_reward: f32, max_reward: f32) -> Self {
        Self {
            min_epsilon,
            max_epsilon,
            min_reward,
            max_reward,
            eps_range: max_epsilon - min_epsilon,
        }
    }
}

#[derive(Clone)]
pub struct EpsilonDecreasing {
    final_epsilon: f32,
    epsilon_decay: Rc<dyn Fn(f32) -> f32>,
}

impl EpsilonDecreasing {
    pub fn new(final_epsilon: f32, epsilon_decay: Rc<dyn Fn(f32) -> f32>) -> Self {
        Self {
            final_epsilon,
            epsilon_decay,
        }
    }
}

pub enum EpsilonUpdateStrategy {
    AdaptativeEpsilon(AdaptativeEpsilon),
    EpsilonDecreasing(EpsilonDecreasing),
    None,
}

impl EpsilonUpdateStrategy {
    fn update(&mut self, current_epsilon: f32, epi_reward: f32) -> f32 {
        match self {
            EpsilonUpdateStrategy::AdaptativeEpsilon(data) => {
                if epi_reward < data.min_reward {
                    data.max_epsilon
                } else {
                    let reward_range = data.max_reward - data.min_reward;
                    let min_update = data.eps_range / reward_range;
                    let new_eps = (data.max_reward - epi_reward) * min_update;
                    if new_eps < data.min_epsilon {
                        data.min_epsilon
                    } else {
                        new_eps
                    }
                }
            }
            EpsilonUpdateStrategy::EpsilonDecreasing(data) => {
                let new_epsilon: f32 = (data.epsilon_decay)(current_epsilon);

                if data.final_epsilon > new_epsilon {
                    current_epsilon
                } else {
                    new_epsilon
                }
            }
            EpsilonUpdateStrategy::None => current_epsilon,
        }
    }
}

pub struct EpsilonGreedy {
    epsilon: f32,
    rng: Rng,
    update_strategy: EpsilonUpdateStrategy,
}

impl Default for EpsilonGreedy {
    fn default() -> Self {
        Self::new(0.1, 42, EpsilonUpdateStrategy::None)
    }
}

impl EpsilonGreedy {
    pub fn new(epsilon: f32, seed: u64, update_strategy: EpsilonUpdateStrategy) -> Self {
        Self {
            epsilon,
            rng: Rng::with_seed(seed),
            update_strategy,
        }
    }

    fn should_explore(&mut self) -> bool {
        self.epsilon != 0.0 && self.rng.f32() <= self.epsilon
    }
    fn get_action(&mut self, values: &Array1<f32>) -> usize {
        if self.should_explore() {
            self.rng.usize(0..values.len())
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

    pub fn get_epsilon(&self) -> f32 {
        self.epsilon
    }
}

impl DiscreteObsDiscreteActionSelection for EpsilonGreedy {
    fn get_action(&mut self, _obs: usize, values: &Array1<f32>) -> usize {
        self.get_action(values)
    }

    fn get_exploration_probs(&mut self, _obs: usize, values: &Array1<f32>) -> Array1<f32> {
        self.get_exploration_probs(values)
    }

    fn reset(&mut self) {}

    fn update(&mut self, epi_reward: f32) {
        self.epsilon = self.update_strategy.update(self.epsilon, epi_reward)
    }
}

impl ContinuousObsDiscreteActionSelection for EpsilonGreedy {
    fn get_action(&mut self, values: &Array1<f32>) -> usize {
        self.get_action(values)
    }

    fn get_exploration_probs(&mut self, values: &Array1<f32>) -> Array1<f32> {
        self.get_exploration_probs(values)
    }

    fn reset(&mut self) {}

    fn update(&mut self, epi_reward: f32) {
        self.epsilon = self.update_strategy.update(self.epsilon, epi_reward)
    }
}
