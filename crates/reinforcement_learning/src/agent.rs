// mod elegibility_traces_agent;
// mod internal_model_agent;
// mod one_step_agent;

// pub use elegibility_traces_agent::ElegibilityTracesAgent;
// pub use internal_model_agent::InternalModelAgent;
// pub use one_step_agent::OneStepAgent;
pub mod one_step_epsilon_greed_sarsa;
pub mod one_step_qlearning;

extern crate environments;

// pub fn expected_sarsa<A: DiscreteAction>(
//     next_q_values: &[f64; A::RANGE],
//     _next_action: A,
//     policy_probs: &[f64; A::RANGE],
// ) -> f64 {
//     let mut future_q_value: f64 = 0.0;
//     for i in 0..next_q_values.len() {
//         future_q_value += policy_probs[i] * next_q_values[i]
//     }
//     future_q_value
// }

pub type TrainResults = (Vec<f64>, Vec<u128>, Vec<f64>, Vec<f64>, Vec<f64>);
pub trait DiscreteAgent {
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
