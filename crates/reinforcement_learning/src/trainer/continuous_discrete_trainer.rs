use std::io::{self, BufRead};

use environments::{env::DiscreteActionEnv, space::SpaceType};
use tch::Tensor;

use crate::agent::CDAgent;

use super::TrainResults;

pub struct CDTrainer<EnvError> {
    env: Box<dyn DiscreteActionEnv<Error = EnvError>>,
    pub early_stop: Option<Box<dyn Fn(f32) -> bool>>,
}

impl<EnvError> CDTrainer<EnvError> {
    pub fn new(env: Box<dyn DiscreteActionEnv<Error = EnvError>>) -> Self {
        if env.observation_space().get_type() != SpaceType::Continuous {
            panic!("observation_space must be of type SpaceType::Continuous");
        }
        if env.action_space().get_type() != SpaceType::Discrete {
            panic!("action_space must be of type SpaceType::Discrete");
        }
        Self {
            env,
            early_stop: None,
        }
    }

    pub fn train_by_steps(
        &mut self,
        agent: &mut dyn CDAgent,
        n_steps: u128,
        update_freq: u128,
        eval_at: u128,
        eval_for: u128,
        debug: bool,
    ) -> Result<TrainResults, EnvError> {
        let mut training_reward: Vec<f32> = vec![];
        let mut training_length: Vec<u128> = vec![];
        let mut training_error: Vec<f32> = vec![];
        let mut evaluation_reward: Vec<f32> = vec![];
        let mut evaluation_length: Vec<f32> = vec![];

        let mut n_episodes = 0;
        let mut action_counter: u128 = 0;
        let mut epi_reward: f32 = 0.0;
        let mut curr_obs: Tensor = self.env.reset()?.try_into().unwrap();
        let mut curr_action = agent.get_action(&curr_obs);

        for _ in 0..n_steps {
            let (next_obs, reward, terminated) = self.env.step(curr_action)?;
            let next_obs = next_obs.try_into().unwrap();

            epi_reward += reward;

            if debug {
                println!("{}", self.env.render());
                let mut line = String::new();
                let stdin = io::stdin();
                stdin.lock().read_line(&mut line).unwrap();
            }

            let next_action: usize = agent.get_action(&next_obs);

            agent.add_transition(
                &curr_obs,
                curr_action,
                reward,
                terminated,
                &next_obs,
                next_action,
            );

            curr_obs = next_obs;
            curr_action = next_action;

            if let Some(td) = agent.update() {
                training_error.push(td)
            }
            if terminated {
                if debug {
                    println!("{}", self.env.render());
                }
                training_reward.push(epi_reward);

                if n_episodes % update_freq == 0 && agent.update_networks().is_err() {
                    println!("copy error")
                }
                if n_episodes % eval_at == 0 {
                    let (rewards, eval_lengths) = self.evaluate(agent, eval_for)?;
                    let reward_avg = (rewards.iter().sum::<f32>()) / (rewards.len() as f32);
                    let eval_lengths_avg = (eval_lengths.iter().map(|x| *x as f32).sum::<f32>())
                        / (eval_lengths.len() as f32);
                    println!("Episode: {}, Avg Return: {:.3} ", n_episodes, reward_avg,);
                    evaluation_reward.push(reward_avg);
                    evaluation_length.push(eval_lengths_avg);
                    if let Some(s) = &self.early_stop {
                        println!("has check");
                        if (s)(reward_avg) {
                            println!("end!");
                            break;
                        };
                    }
                }
                curr_obs = self.env.reset()?.try_into().unwrap();
                agent.action_selection_update(epi_reward);

                n_episodes += 1;
                epi_reward = 0.0;
                action_counter = 0;
            }
            training_length.push(action_counter);
        }
        Ok((
            training_reward,
            training_length,
            training_error,
            evaluation_reward,
            evaluation_length,
        ))
    }

    pub fn train_by_steps2(
        &mut self,
        agent: &mut dyn CDAgent,
        n_steps: u128,
        update_freq: u128,
        eval_at: u128,
        eval_for: u128,
        debug: bool,
    ) -> Result<TrainResults, EnvError> {
        let mut training_reward: Vec<f32> = vec![];
        let mut training_length: Vec<u128> = vec![];
        let mut training_error: Vec<f32> = vec![];
        let mut evaluation_reward: Vec<f32> = vec![];
        let mut evaluation_length: Vec<f32> = vec![];

        let mut n_episodes = 0;
        let mut action_counter: u128 = 0;
        let mut epi_reward: f32 = 0.0;
        let mut curr_obs: Tensor = self.env.reset()?.try_into().unwrap();

        for _ in 0..n_steps {
            let curr_action = agent.get_action(&curr_obs);

            let (next_obs, reward, terminated) = self.env.step(curr_action)?;
            let next_obs = next_obs.try_into().unwrap();
            epi_reward += reward;
            agent.add_transition(
                &curr_obs,
                curr_action,
                reward,
                terminated,
                &next_obs,
                curr_action, // HERE
            );

            curr_obs = next_obs;

            if let Some(td) = agent.update() {
                training_error.push(td)
            }

            if terminated {
                if debug {
                    println!("{}", self.env.render());
                }
                training_reward.push(epi_reward);
                if n_episodes % update_freq == 0 && agent.update_networks().is_err() {
                    println!("copy error")
                }
                if n_episodes % eval_at == 0 {
                    let (rewards, eval_lengths) = self.evaluate(agent, eval_for)?;
                    let reward_avg = (rewards.iter().sum::<f32>()) / (rewards.len() as f32);
                    let eval_lengths_avg = (eval_lengths.iter().map(|x| *x as f32).sum::<f32>())
                        / (eval_lengths.len() as f32);
                    println!("Episode: {}, Avg Return: {:.3} ", n_episodes, reward_avg,);
                    evaluation_reward.push(reward_avg);
                    evaluation_length.push(eval_lengths_avg);
                    if let Some(s) = &self.early_stop {
                        if (s)(reward_avg) {
                            break;
                        };
                    }
                }
                curr_obs = self.env.reset()?.try_into().unwrap();
                agent.action_selection_update(epi_reward);
                n_episodes += 1;
                epi_reward = 0.0;
                action_counter = 0;
            }
            training_length.push(action_counter);
        }
        Ok((
            training_reward,
            training_length,
            training_error,
            evaluation_reward,
            evaluation_length,
        ))
    }

    pub fn evaluate(
        &mut self,
        agent: &mut dyn CDAgent,
        n_episodes: u128,
    ) -> Result<(Vec<f32>, Vec<u128>), EnvError> {
        let mut reward_history: Vec<f32> = vec![];
        let mut episode_length: Vec<u128> = vec![];
        for _episode in 0..n_episodes {
            let mut action_counter: u128 = 0;
            let mut epi_reward: f32 = 0.0;
            let obs_repr = self.env.reset()?.try_into().unwrap();
            let mut curr_action = agent.get_best_action(&obs_repr);
            loop {
                action_counter += 1;
                let (obs, reward, terminated) = self.env.step(curr_action)?;
                let next_obs_repr = obs.try_into().unwrap();
                let next_action_repr: usize = agent.get_best_action(&next_obs_repr);
                let next_action = next_action_repr;
                curr_action = next_action;
                epi_reward += reward;
                if terminated {
                    reward_history.push(epi_reward);
                    break;
                }
                if let Some(s) = &self.early_stop {
                    if (s)(epi_reward) {
                        reward_history.push(epi_reward);
                        break;
                    };
                }
            }
            episode_length.push(action_counter);
        }
        Ok((reward_history, episode_length))
    }

    pub fn example(
        &mut self,
        env: &mut impl DiscreteActionEnv<Error = EnvError>,
        agent: &mut impl CDAgent,
    ) -> Result<(), EnvError> {
        let mut epi_reward = 0.0;
        let obs_repr = env.reset()?.try_into().unwrap();
        let mut curr_action = agent.get_action(&obs_repr);
        let mut steps: i32 = 0;
        loop {
            steps += 1;
            println!("{}", env.render());
            let (next_obs, reward, terminated) = env.step(curr_action)?;
            let next_obs_repr = next_obs.try_into().unwrap();
            let next_action_repr: usize = agent.get_action(&next_obs_repr);
            let next_action = next_action_repr;
            println!("action : {}", next_action_repr);
            println!("step reward {:?}", reward);
            curr_action = next_action;
            epi_reward += reward;
            if terminated {
                println!("{}", env.render());
                println!("episode reward {:?}", epi_reward);
                println!("terminated with {:?} steps", steps);
                break Ok(());
            }
        }
    }
}
