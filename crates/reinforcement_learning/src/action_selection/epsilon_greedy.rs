use super::{ContinuousObsDiscreteActionSelection, DiscreteObsDiscreteActionSelection};
use ndarray::{Array, Array1};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use utils::argmax;

pub enum EpsilonUpdateStrategy {
    AdaptativeEpsilon {
        min_epsilon: f32,
        max_epsilon: f32,
        min_reward: f32,
        max_reward: f32,
        eps_range: f32,
    },
    EpsilonDecreasing {
        final_epsilon: f32,
        epsilon_decay: Box<dyn Fn(f32) -> f32 + Send + Sync>,
    },
    None,
}

impl EpsilonUpdateStrategy {
    fn update(&mut self, current_epsilon: f32, epi_reward: f32) -> f32 {
        match self {
            EpsilonUpdateStrategy::AdaptativeEpsilon {
                min_epsilon,
                max_epsilon,
                min_reward,
                max_reward,
                eps_range,
            } => {
                if epi_reward < *min_reward {
                    *max_epsilon
                } else {
                    let reward_range = *max_reward - *min_reward;
                    let min_update = *eps_range / reward_range;
                    let new_eps = (*max_reward - epi_reward) * min_update;
                    if new_eps < *min_epsilon {
                        *min_epsilon
                    } else {
                        new_eps
                    }
                }
            }
            EpsilonUpdateStrategy::EpsilonDecreasing {
                final_epsilon,
                epsilon_decay,
            } => {
                let new_epsilon: f32 = (epsilon_decay)(current_epsilon);

                if *final_epsilon > new_epsilon {
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
    rng: SmallRng,
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
            rng: SmallRng::seed_from_u64(seed),
            update_strategy,
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
