use std::cell::{RefCell, RefMut};

use crate::policy::Policy;

mod one_step_qlearning;
mod one_step_sarsa;
mod qlearning_lambda;
mod sarsa_lambda;
mod one_step_expected_sarsa;

pub use one_step_qlearning::OneStepQLearning;
pub use one_step_sarsa::OneStepSARSA;
pub use qlearning_lambda::QLearningLambda;
pub use sarsa_lambda::SarsaLambda;
pub use one_step_expected_sarsa::OneStepExpectedSarsa;

use super::action_selection::ActionSelection;


pub trait PolicyUpdate<T> {
    fn update(
        &mut self,
        curr_obs:T,
        curr_action: usize,
        next_obs: T,
        next_action: usize,
        reward: f64,
        terminated: bool,
        policy: RefMut<&mut dyn Policy<T>>,
        action_selection: &Box<RefCell<&mut dyn ActionSelection<T>>>
    ) -> f64;
}