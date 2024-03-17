use std::io::{self, BufRead};

use environments::{env::Env, space::SpaceType};
use tch::Tensor;

use crate::agent::ContinuousObsDiscreteActionAgent;

use super::TrainResults;

pub struct ContinuousObsDiscreteTrainer<Obs, Action> {
    train_env: Box<dyn Env<Obs, Action>>,
    eval_env: Box<dyn Env<Obs, Action>>,
    pub repr_to_action: fn(usize) -> Action,
    pub obs_to_repr: fn(&Obs) -> Tensor,
    // reward_observers: Vec<Box<dyn Subscriber<f32>>>,
}

impl<Obs, Action> ContinuousObsDiscreteTrainer<Obs, Action> {
    pub fn new(
        train_env: Box<dyn Env<Obs, Action>>,
        eval_env: Box<dyn Env<Obs, Action>>,
        repr_to_action: fn(usize) -> Action,
        obs_to_repr: fn(&Obs) -> Tensor,
    ) -> Self {
        if train_env.observation_space().get_type() != SpaceType::Continuous {
            panic!("observation_space must be of type SpaceType::Continuous");
        }
        if train_env.action_space().get_type() != SpaceType::Discrete {
            panic!("action_space must be of type SpaceType::Discrete");
        }
        if eval_env.observation_space().get_type() != SpaceType::Continuous {
            panic!("observation_space must be of type SpaceType::Continuous");
        }
        if eval_env.action_space().get_type() != SpaceType::Discrete {
            panic!("action_space must be of type SpaceType::Discrete");
        }
        Self {
            train_env,
            eval_env,
            repr_to_action,
            obs_to_repr,
            // reward_observers: vec![],
        }
    }

    pub fn train_by_steps(
        &mut self,
        agent: &mut dyn ContinuousObsDiscreteActionAgent,
        n_steps: u128,
        update_freq: u128,
        eval_at: u128,
        eval_for: u128,
        debug: bool,
    ) -> TrainResults
    where
        Obs: Clone,
    {
        let mut training_reward: Vec<f32> = vec![];
        let mut training_length: Vec<u128> = vec![];
        let mut training_error: Vec<f32> = vec![];
        let mut evaluation_reward: Vec<f32> = vec![];
        let mut evaluation_length: Vec<f32> = vec![];

        let mut n_episodes = 0;
        let mut action_counter: u128 = 0;
        let mut epi_reward: f32 = 0.0;
        let mut curr_obs: Tensor = (self.obs_to_repr)(&self.train_env.reset());
        let mut curr_action = agent.get_action(&curr_obs);

        for _ in 0..n_steps {
            let (next_obs, reward, terminated) = self
                .train_env
                .step((self.repr_to_action)(curr_action))
                .unwrap();
            let next_obs = (self.obs_to_repr)(&next_obs);

            epi_reward += reward;

            if debug {
                println!("{}", self.train_env.render());
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
                    println!("{}", self.train_env.render());
                }
                training_reward.push(epi_reward);
                // self.emit_reward_signal(epi_reward);

                if n_episodes % update_freq == 0 && agent.update_networks().is_err() {
                    println!("copy error")
                }
                if n_episodes % eval_at == 0 {
                    let (rewards, eval_lengths) = self.evaluate(agent, eval_for);
                    let reward_avg = (rewards.iter().sum::<f32>()) / (rewards.len() as f32);
                    let eval_lengths_avg = (eval_lengths.iter().map(|x| *x as f32).sum::<f32>())
                        / (eval_lengths.len() as f32);
                    println!("Episode: {}, Avg Return: {:.3} ", n_episodes, reward_avg,);
                    evaluation_reward.push(reward_avg);
                    evaluation_length.push(eval_lengths_avg);
                }
                curr_obs = (self.obs_to_repr)(&self.train_env.reset());
                agent.action_selection_update(epi_reward);

                n_episodes += 1;
                epi_reward = 0.0;
                action_counter = 0;
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

    pub fn train_by_steps2(
        &mut self,
        agent: &mut dyn ContinuousObsDiscreteActionAgent,
        n_steps: u128,
        update_freq: u128,
        eval_at: u128,
        eval_for: u128,
        debug: bool,
    ) -> TrainResults
    where
        Obs: Clone,
    {
        let mut training_reward: Vec<f32> = vec![];
        let mut training_length: Vec<u128> = vec![];
        let mut training_error: Vec<f32> = vec![];
        let mut evaluation_reward: Vec<f32> = vec![];
        let mut evaluation_length: Vec<f32> = vec![];

        let mut n_episodes = 0;
        let mut action_counter: u128 = 0;
        let mut epi_reward: f32 = 0.0;
        let mut curr_obs: Tensor = (self.obs_to_repr)(&self.train_env.reset());

        for _ in 0..n_steps {
            let curr_action = agent.get_action(&curr_obs);

            let (next_obs, reward, terminated) = self
                .train_env
                .step((self.repr_to_action)(curr_action))
                .unwrap();
            let next_obs = (self.obs_to_repr)(&next_obs);

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
                    println!("{}", self.train_env.render());
                }
                training_reward.push(epi_reward);
                if n_episodes % update_freq == 0 && agent.update_networks().is_err() {
                    println!("copy error")
                }
                if n_episodes % eval_at == 0 {
                    let (rewards, eval_lengths) = self.evaluate(agent, eval_for);
                    let reward_avg = (rewards.iter().sum::<f32>()) / (rewards.len() as f32);
                    let eval_lengths_avg = (eval_lengths.iter().map(|x| *x as f32).sum::<f32>())
                        / (eval_lengths.len() as f32);
                    println!("Episode: {}, Avg Return: {:.3} ", n_episodes, reward_avg,);
                    evaluation_reward.push(reward_avg);
                    evaluation_length.push(eval_lengths_avg);
                }
                curr_obs = (self.obs_to_repr)(&self.train_env.reset());
                agent.action_selection_update(epi_reward);

                n_episodes += 1;
                epi_reward = 0.0;
                action_counter = 0;
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

    pub fn evaluate(
        &mut self,
        agent: &mut dyn ContinuousObsDiscreteActionAgent,
        n_episodes: u128,
    ) -> (Vec<f32>, Vec<u128>) {
        let mut reward_history: Vec<f32> = vec![];
        let mut episode_length: Vec<u128> = vec![];
        for _episode in 0..n_episodes {
            let mut action_counter: u128 = 0;
            let mut epi_reward: f32 = 0.0;
            let binding = self.eval_env.reset();
            let obs_repr = (self.obs_to_repr)(&binding);
            let action_repr: usize = agent.get_best_action(&obs_repr);
            let mut curr_action = (self.repr_to_action)(action_repr);
            loop {
                action_counter += 1;
                let (obs, reward, terminated) = self.eval_env.step(curr_action).unwrap();
                let next_obs_repr = (self.obs_to_repr)(&obs);
                let next_action_repr: usize = agent.get_best_action(&next_obs_repr);
                let next_action = (self.repr_to_action)(next_action_repr);
                curr_action = next_action;
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

    pub fn example(
        &mut self,
        env: &mut impl Env<Obs, Action>,
        agent: &mut impl ContinuousObsDiscreteActionAgent,
    ) {
        let mut epi_reward = 0.0;
        let binding = env.reset();
        let obs_repr = (self.obs_to_repr)(&binding);
        let action_repr: usize = agent.get_action(&obs_repr);
        let mut curr_action = (self.repr_to_action)(action_repr);
        let mut steps: i32 = 0;
        loop {
            steps += 1;
            println!("{}", env.render());
            let (next_obs, reward, terminated) = env.step(curr_action).unwrap();
            let next_obs_repr = (self.obs_to_repr)(&next_obs);
            let next_action_repr: usize = agent.get_action(&next_obs_repr);
            let next_action = (self.repr_to_action)(next_action_repr);
            println!("action : {}", next_action_repr);
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
