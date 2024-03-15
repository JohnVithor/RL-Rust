mod elegibility_traces_agent;
// mod internal_model_agent;
mod double_deep_agent;
mod one_step_agent;

use ndarray::Array1;

pub use elegibility_traces_agent::ElegibilityTracesAgent;
// pub use internal_model_agent::InternalModelAgent;
pub use double_deep_agent::DoubleDeepAgent;
pub use double_deep_agent::OptimizerEnum;
pub use one_step_agent::OneStepAgent;
use tch::{TchError, Tensor};

extern crate environments;

pub type GetNextQValue = fn(&Array1<f32>, usize, &Array1<f32>) -> f32;
// epsilon_decay: Rc<dyn Fn(GetNextQValue) -> f32>,

pub fn sarsa(next_q_values: &Array1<f32>, next_action: usize, _policy_probs: &Array1<f32>) -> f32 {
    next_q_values[next_action]
}

pub fn qlearning(
    next_q_values: &Array1<f32>,
    _next_action: usize,
    _policy_probs: &Array1<f32>,
) -> f32 {
    next_q_values.iter().fold(f32::NAN, |acc, x| acc.max(*x))
}

pub fn expected_sarsa(
    next_q_values: &Array1<f32>,
    _next_action: usize,
    policy_probs: &Array1<f32>,
) -> f32 {
    let mut future_q_value: f32 = 0.0;
    for i in 0..next_q_values.len() {
        future_q_value += policy_probs[i] * next_q_values[i]
    }
    future_q_value
}

pub type TrainResults = (Vec<f32>, Vec<u128>, Vec<f32>, Vec<f32>, Vec<f32>);
pub trait FullDiscreteAgent {
    fn prepare(&mut self, n_obs: usize, n_actions: usize);

    fn get_action(&mut self, obs: usize) -> usize;

    fn update(
        &mut self,
        curr_obs: usize,
        curr_action: usize,
        reward: f32,
        terminated: bool,
        next_obs: usize,
        next_action: usize,
    ) -> f32;

    fn reset(&mut self);
}

pub trait ContinuousObsDiscreteActionAgent {
    fn get_action(&mut self, state: &Tensor) -> usize;

    fn action_selection_update(&mut self, epi_reward: f32);

    fn get_best_action(&self, state: &Tensor) -> usize;

    fn add_transition(
        &mut self,
        curr_state: &Tensor,
        curr_action: usize,
        reward: f32,
        done: bool,
        next_state: &Tensor,
        next_action: usize,
    );

    fn update_networks(&mut self) -> Result<(), TchError>;

    fn get_batch(&mut self, size: usize) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor);

    fn batch_qvalues(&self, b_states: &Tensor, b_actions: &Tensor) -> Tensor;

    fn batch_expected_values(
        &self,
        b_state_: &Tensor,
        b_reward: &Tensor,
        b_done: &Tensor,
    ) -> Tensor;

    fn episode_end_hook(&mut self) {}

    fn optimize(&mut self, loss: &Tensor);

    fn update(&mut self) -> Option<f32>;

    fn reset(&mut self);
}

pub trait FullContinuousAgent {
    // fn prepare(&mut self, n_obs: usize, n_actions: usize);

    fn get_action(&mut self, obs: &Array1<f32>) -> Array1<f32>;

    fn update(
        &mut self,
        curr_obs: &Array1<f32>,
        curr_action: &Array1<f32>,
        reward: f32,
        terminated: bool,
        next_obs: &Array1<f32>,
        next_action: &Array1<f32>,
    ) -> f32;

    fn reset(&mut self);
}
