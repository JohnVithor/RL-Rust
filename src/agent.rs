use std::fmt::Debug;
use std::hash::Hash;

use crate::env::Env;
use crate::utils::max;

pub type GetNextQValue<const COUNT: usize> = fn(&[f64; COUNT], usize, f64) -> f64;

pub fn sarsa<const COUNT: usize>(
    next_q_values: &[f64; COUNT],
    next_action: usize,
    _epsilon: f64,
) -> f64 {
    return next_q_values[next_action];
}

pub fn qlearning<const COUNT: usize>(
    next_q_values: &[f64; COUNT],
    _next_action: usize,
    _epsilon: f64,
) -> f64 {
    return max(next_q_values);
}

pub fn expected_sarsa<const COUNT: usize>(
    next_q_values: &[f64; COUNT],
    _next_action: usize,
    epsilon: f64,
) -> f64 {
    let policy_probs: [f64; COUNT] = [epsilon / COUNT as f64; COUNT];
    let best_action_value: f64 = max(next_q_values);

    let mut n_max_action: i32 = 0;
    for i in 0..COUNT {
        if next_q_values[i] == best_action_value {
            n_max_action += 1;
        }
    }
    let mut future_q_value: f64 = 0.0;
    for i in 0..COUNT {
        if next_q_values[i] == best_action_value {
            future_q_value +=
                (policy_probs[i] + (1.0 - epsilon) / n_max_action as f64) * next_q_values[i]
        } else {
            future_q_value += policy_probs[i] * next_q_values[i]
        }
    }
    return future_q_value;
}

pub trait Agent<T: Hash + PartialEq + Eq + Clone + Debug, const COUNT: usize> {
    fn get_action(&mut self, obs: &T) -> usize;

    fn update(
        &mut self,
        curr_obs: &T,
        curr_action: usize,
        reward: f64,
        terminated: bool,
        next_obs: &T,
        next_action: usize,
    );

    fn get_training_error(&self) -> &Vec<f64>;

    fn reset(&mut self);

    fn train(&mut self, env: &mut dyn Env<T, COUNT>, n_episodes: u128) -> (Vec<f64>, Vec<u128>) {
        let mut reward_history: Vec<f64> = vec![];
        let mut episode_length: Vec<u128> = vec![];
        for _episode in 0..n_episodes {
            let mut action_counter: u128 = 0;
            let mut epi_reward: f64 = 0.0;
            let mut curr_obs: T = env.reset();
            let mut curr_action: usize = self.get_action(&curr_obs);
            loop {
                action_counter += 1;
                let (next_obs, reward, terminated) = env.step(curr_action).unwrap();
                let next_action: usize = self.get_action(&next_obs);
                self.update(
                    &curr_obs,
                    curr_action,
                    reward,
                    terminated,
                    &next_obs,
                    next_action,
                );
                curr_obs = next_obs;
                curr_action = next_action;
                epi_reward += reward;
                if terminated {
                    reward_history.push(epi_reward);
                    break;
                }
            }
            episode_length.push(action_counter);
        }
        return (reward_history, episode_length);
    }

    fn example(&mut self, env: &mut dyn Env<T, COUNT>) {
        let mut epi_reward = 0.0;
        let mut curr_action: usize = self.get_action(&env.reset());
        let mut steps: i32 = 0;
        loop {
            steps += 1;
            println!("{}", env.render());
            let (next_obs, reward, terminated) = env.step(curr_action).unwrap();
            let next_action: usize = self.get_action(&next_obs);
            println!("{:?}", env.get_action_label(curr_action));
            println!("step reward {:?}", reward);
            curr_action = next_action;
            epi_reward += reward;
            if terminated {
                println!("{}", env.render());
                println!("episode reward {:?}", epi_reward);
                println!("terminated with {:?} steps", steps);
                break;
            }
        }
    }
}

mod elegibility_traces_tabular_egreedy_agent;
mod one_step_tabular_egreedy_agent;
mod one_step_tabular_egreedy_double_agent;
mod one_step_tabular_ucb_agent;

pub use elegibility_traces_tabular_egreedy_agent::ElegibilityTracesTabularEGreedyAgent;
pub use one_step_tabular_egreedy_agent::OneStepTabularEGreedyAgent;
pub use one_step_tabular_egreedy_double_agent::OneStepTabularEGreedyDoubleAgent;
pub use one_step_tabular_ucb_agent::OneStepTabularUCBAgent;