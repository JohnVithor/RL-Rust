mod elegibility_traces_agent;
// mod internal_model_agent;
mod one_step_agent;

use ndarray::Array1;

pub use elegibility_traces_agent::ElegibilityTracesAgent;
// pub use internal_model_agent::InternalModelAgent;
pub use one_step_agent::OneStepAgent;

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
pub trait DiscreteAgent {
    fn prepare(&mut self, n_obs: usize, n_actions: usize);

    fn get_action(&mut self, obs: usize) -> usize;

    fn update(
        &mut self,
        curr_obs: usize,
        curr_actions: usize,
        reward: f32,
        terminated: bool,
        next_obs: usize,
        next_actions: usize,
    ) -> f32;

    fn reset(&mut self);
}

pub trait ContinuousAgent {
    fn prepare(&mut self, obs_dim: usize, n_actions: usize);

    fn get_action(&mut self, obs: Array1<f32>) -> usize;

    fn update(
        &mut self,
        curr_obs: Array1<f32>,
        curr_actions: usize,
        reward: f32,
        terminated: bool,
        next_obs: Array1<f32>,
        next_actions: usize,
    ) -> f32;

    fn reset(&mut self);
}
