// mod elegibility_traces_agent;
// mod internal_model_agent;
// mod one_step_agent;

use ndarray::Array1;

// pub use elegibility_traces_agent::ElegibilityTracesAgent;
// pub use internal_model_agent::InternalModelAgent;
// pub use one_step_agent::OneStepAgent;
pub mod one_step_agent;

extern crate environments;

pub type GetNextQValue = fn(&Array1<f64>, usize, &Array1<f64>) -> f64;
// epsilon_decay: Rc<dyn Fn(GetNextQValue) -> f64>,

pub fn sarsa(next_q_values: &Array1<f64>, next_action: usize, _policy_probs: &Array1<f64>) -> f64 {
    next_q_values[next_action]
}

pub fn qlearning(
    next_q_values: &Array1<f64>,
    _next_action: usize,
    _policy_probs: &Array1<f64>,
) -> f64 {
    next_q_values.iter().fold(f64::NAN, |acc, x| acc.max(*x))
}

pub fn expected_sarsa(
    next_q_values: &Array1<f64>,
    _next_action: usize,
    policy_probs: &Array1<f64>,
) -> f64 {
    let mut future_q_value: f64 = 0.0;
    for i in 0..next_q_values.len() {
        future_q_value += policy_probs[i] * next_q_values[i]
    }
    future_q_value
}

pub type TrainResults = (Vec<f64>, Vec<u128>, Vec<f64>, Vec<f64>, Vec<f64>);
pub trait DiscreteAgent {
    fn prepare(&mut self, n_obs: usize, n_actions: usize);
    fn get_action(&mut self, obs: usize) -> usize;

    fn update(
        &mut self,
        curr_obs: usize,
        curr_actions: usize,
        reward: f64,
        terminated: bool,
        next_obs: usize,
        next_actions: usize,
    ) -> f64;

    fn reset(&mut self);
}
