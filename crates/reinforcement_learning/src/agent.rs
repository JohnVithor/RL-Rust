mod elegibility_traces_agent;
mod internal_model_agent;
mod one_step_agent;

pub use elegibility_traces_agent::ElegibilityTracesAgent;
// pub use internal_model_agent::InternalModelAgent;
pub use one_step_agent::OneStepAgent;

use crate::utils::max;
use crate::{action_selection::ActionSelection, policy::DiscretePolicy};
use environments::env::DiscreteAction;
use std::{fmt::Debug, ops::Index};

extern crate environments;
use environments::env::DiscreteEnv;

pub fn sarsa<A: DiscreteAction + From<usize>>(
    next_q_values: &[f64; A::RANGE],
    next_action: A,
    _policy_probs: &[f64; A::RANGE],
) -> f64
where
    [f64]: Index<A, Output = f64>,
{
    next_q_values[next_action]
}

pub fn qlearning<A: DiscreteAction>(
    next_q_values: &[f64; A::RANGE],
    _next_action: A,
    _policy_probs: &[f64; A::RANGE],
) -> f64 {
    max(next_q_values)
}

pub fn expected_sarsa<A: DiscreteAction>(
    next_q_values: &[f64; A::RANGE],
    _next_action: A,
    policy_probs: &[f64; A::RANGE],
) -> f64 {
    let mut future_q_value: f64 = 0.0;
    for i in 0..next_q_values.len() {
        future_q_value += policy_probs[i] * next_q_values[i]
    }
    future_q_value
}

pub type TrainResults = (Vec<f64>, Vec<u128>, Vec<f64>, Vec<f64>, Vec<f64>);
pub trait DiscreteAgent<'a, T: Clone + Debug, A: DiscreteAction + Debug + Copy>
where
    [(); A::RANGE]: Sized,
{
    fn get_policy(&self) -> &dyn DiscretePolicy<T, A>;

    fn set_future_q_value_func(&mut self, func: fn(&[f64; A::RANGE], A, &[f64; A::RANGE]) -> f64);

    fn set_action_selector(&mut self, action_selector: &'a mut dyn ActionSelection<T, A>);

    fn get_action(&mut self, obs: &T) -> A;

    fn update(
        &mut self,
        curr_obs: &T,
        curr_actions: A,
        reward: f64,
        terminated: bool,
        next_obs: &T,
        next_actions: A,
    ) -> f64;

    fn reset(&mut self);

    fn train(
        &mut self,
        env: &mut dyn DiscreteEnv<T, A>,
        n_episodes: u128,
        eval_at: u128,
        eval_for: u128,
    ) -> TrainResults {
        let mut training_reward: Vec<f64> = vec![];
        let mut training_length: Vec<u128> = vec![];
        let mut training_error: Vec<f64> = vec![];
        let mut evaluation_reward: Vec<f64> = vec![];
        let mut evaluation_length: Vec<f64> = vec![];

        for episode in 0..n_episodes {
            let mut action_counter: u128 = 0;
            let mut epi_reward: f64 = 0.0;
            let mut curr_obs: T = env.reset();
            let mut curr_actions: A = self.get_action(&curr_obs);

            loop {
                action_counter += 1;
                let (next_obs, reward, terminated) = env.step(curr_actions).unwrap();
                let next_actions: A = self.get_action(&next_obs);
                let td = self.update(
                    &curr_obs,
                    curr_actions,
                    reward,
                    terminated,
                    &next_obs,
                    next_actions,
                );
                training_error.push(td);
                curr_obs = next_obs;
                curr_actions = next_actions;
                epi_reward += reward;
                if terminated {
                    training_reward.push(epi_reward);
                    break;
                }
            }
            if episode % eval_at == 0 {
                let (r, l) = self.evaluate(env, eval_for);
                let mr: f64 = r.iter().sum::<f64>() / r.len() as f64;
                let ml: f64 = l.iter().sum::<u128>() as f64 / l.len() as f64;
                evaluation_reward.push(mr);
                evaluation_length.push(ml);
            }
            training_length.push(action_counter);
        }
        (
            training_reward,
            training_length,
            training_error,
            evaluation_reward,
            evaluation_length,
        )
    }

    fn evaluate(
        &mut self,
        env: &mut dyn DiscreteEnv<T, A>,
        n_episodes: u128,
    ) -> (Vec<f64>, Vec<u128>) {
        let mut reward_history: Vec<f64> = vec![];
        let mut episode_length: Vec<u128> = vec![];
        for _episode in 0..n_episodes {
            let mut action_counter: u128 = 0;
            let mut epi_reward: f64 = 0.0;
            let mut curr_actions: A = self.get_action(&env.reset());
            loop {
                action_counter += 1;
                let (obs, reward, terminated) = env.step(curr_actions).unwrap();
                let next_actions: A = self.get_action(&obs);
                curr_actions = next_actions;
                epi_reward += reward;
                if terminated {
                    reward_history.push(epi_reward);
                    break;
                }
            }
            episode_length.push(action_counter);
        }
        (reward_history, episode_length)
    }

    fn example(&mut self, env: &mut dyn DiscreteEnv<T, A>) {
        let mut epi_reward = 0.0;
        let mut curr_actions: A = self.get_action(&env.reset());
        let mut steps: i32 = 0;
        loop {
            steps += 1;
            println!("{}", env.render());
            let (next_obs, reward, terminated) = env.step(curr_actions).unwrap();
            let next_actions: A = self.get_action(&next_obs);
            // println!("{:?}", env.get_action_label(curr_action));
            println!("step reward {:?}", reward);
            curr_actions = next_actions;
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
