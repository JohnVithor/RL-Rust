use std::cell::RefCell;

use crate::Policy;

mod q_step;
mod sarsa_step;
mod q_lambda;
mod sarsa_lambda;
mod expected_sarsa_step;

pub use q_step::QStep;
pub use sarsa_step::SarsaStep;
pub use q_lambda::QLambda;
pub use sarsa_lambda::SarsaLambda;
pub use expected_sarsa_step::ExpectedSarsaStep;

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
        policy: &mut Policy<T>,
        action_selection: &Box<RefCell<dyn ActionSelection<T>>>
    ) -> f64;
}