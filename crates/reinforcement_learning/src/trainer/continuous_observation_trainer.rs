use std::io::{self, BufRead};

use environments::{env::Env, space::SpaceType};
use ndarray::Array1;

use crate::agent::ContinuousObsDiscreteActionAgent;

use super::TrainResults;

pub struct ContinuousObsDiscreteTrainer<Obs, Action> {
    pub repr_to_action: fn(usize) -> Action,
    pub obs_to_repr: fn(&Obs) -> Array1<f32>,
}

impl<Obs, Action> ContinuousObsDiscreteTrainer<Obs, Action> {
    pub fn new(repr_to_action: fn(usize) -> Action, obs_to_repr: fn(&Obs) -> Array1<f32>) -> Self {
        Self {
            repr_to_action,
            obs_to_repr,
        }
    }

    pub fn train_by_episode(
        &mut self,
        env: &mut dyn Env<Obs, Action>,
        agent: &mut dyn ContinuousObsDiscreteActionAgent,
        n_episodes: u128,
        eval_at: u128,
        eval_for: u128,
        debug: bool,
    ) -> TrainResults
    where
        Obs: Clone,
    {
        if env.observation_space().get_type() != SpaceType::Continuous {
            panic!("observation_space must be of type SpaceType::Continuous");
        }
        if env.action_space().get_type() != SpaceType::Discrete {
            panic!("action_space must be of type SpaceType::Discrete");
        }
        let mut training_reward: Vec<f32> = vec![];
        let mut training_length: Vec<u128> = vec![];
        let mut training_error: Vec<f32> = vec![];
        let mut evaluation_reward: Vec<f32> = vec![];
        let mut evaluation_length: Vec<f32> = vec![];

        for episode in 0..n_episodes {
            let mut action_counter: u128 = 0;
            let mut epi_reward: f32 = 0.0;
            let mut curr_obs: Obs = env.reset();
            let mut curr_action_repr: usize = agent.get_action(&(self.obs_to_repr)(&curr_obs));

            loop {
                action_counter += 1;
                let curr_action = (self.repr_to_action)(curr_action_repr);
                let (next_obs, reward, terminated) = env.step(curr_action).unwrap();

                if debug {
                    println!("{}", env.render());
                    let mut line = String::new();
                    let stdin = io::stdin();
                    stdin.lock().read_line(&mut line).unwrap();
                }
                let next_obs_repr = (self.obs_to_repr)(&next_obs);
                let next_action_repr: usize = agent.get_action(&next_obs_repr);
                let td = agent.update(
                    &(self.obs_to_repr)(&curr_obs),
                    curr_action_repr,
                    reward,
                    terminated,
                    &next_obs_repr,
                    next_action_repr,
                );
                training_error.push(td);
                curr_obs = next_obs;
                curr_action_repr = next_action_repr;
                epi_reward += reward;
                if terminated {
                    if debug {
                        println!("{}", env.render());
                    }
                    training_reward.push(epi_reward);
                    break;
                }
            }
            if episode % eval_at == 0 {
                let (r, l) = self.evaluate(env, agent, eval_for);
                let mr: f32 = r.iter().sum::<f32>() / r.len() as f32;
                let ml: f32 = l.iter().sum::<u128>() as f32 / l.len() as f32;
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

    pub fn train_by_steps(
        &mut self,
        env: &mut dyn Env<Obs, Action>,
        agent: &mut dyn ContinuousObsDiscreteActionAgent,
        n_steps: u128,
        eval_at: u128,
        eval_for: u128,
        debug: bool,
    ) -> TrainResults
    where
        Obs: Clone,
    {
        if env.observation_space().get_type() != SpaceType::Continuous {
            panic!("observation_space must be of type SpaceType::Continuous");
        }
        if env.action_space().get_type() != SpaceType::Discrete {
            panic!("action_space must be of type SpaceType::Discrete");
        }
        let mut training_reward: Vec<f32> = vec![];
        let mut training_length: Vec<u128> = vec![];
        let mut training_error: Vec<f32> = vec![];
        let mut evaluation_reward: Vec<f32> = vec![];
        let mut evaluation_length: Vec<f32> = vec![];

        let mut n_episodes = 0;
        let mut action_counter: u128 = 0;
        let mut epi_reward: f32 = 0.0;
        let mut curr_obs: Obs = env.reset();
        let mut curr_action_repr: usize = agent.get_action(&(self.obs_to_repr)(&curr_obs));

        for _ in 0..n_steps {
            let curr_action = (self.repr_to_action)(curr_action_repr);
            let (next_obs, reward, terminated) = env.step(curr_action).unwrap();

            if debug {
                println!("{}", env.render());
                let mut line = String::new();
                let stdin = io::stdin();
                stdin.lock().read_line(&mut line).unwrap();
            }
            let next_obs_repr = (self.obs_to_repr)(&next_obs);
            let next_action_repr: usize = agent.get_action(&next_obs_repr);
            let td = agent.update(
                &(self.obs_to_repr)(&curr_obs),
                curr_action_repr,
                reward,
                terminated,
                &next_obs_repr,
                next_action_repr,
            );
            training_error.push(td);
            curr_obs = next_obs;
            curr_action_repr = next_action_repr;
            epi_reward += reward;
            if terminated {
                if debug {
                    println!("{}", env.render());
                }
                n_episodes += 1;
                training_reward.push(epi_reward);
                epi_reward = 0.0;
                action_counter = 0;
                curr_obs = env.reset();
                if n_episodes % eval_at == 0 {
                    let (r, l) = self.evaluate(env, agent, eval_for);
                    let mr: f32 = r.iter().sum::<f32>() / r.len() as f32;
                    let ml: f32 = l.iter().sum::<u128>() as f32 / l.len() as f32;
                    evaluation_reward.push(mr);
                    evaluation_length.push(ml);
                }
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
        &self,
        env: &mut dyn Env<Obs, Action>,
        agent: &mut dyn ContinuousObsDiscreteActionAgent,
        n_episodes: u128,
    ) -> (Vec<f32>, Vec<u128>) {
        let mut reward_history: Vec<f32> = vec![];
        let mut episode_length: Vec<u128> = vec![];
        for _episode in 0..n_episodes {
            let mut action_counter: u128 = 0;
            let mut epi_reward: f32 = 0.0;
            let binding = env.reset();
            let obs_repr = (self.obs_to_repr)(&binding);
            let action_repr: usize = agent.get_action(&obs_repr);
            let mut curr_action = (self.repr_to_action)(action_repr);
            loop {
                action_counter += 1;
                let (obs, reward, terminated) = env.step(curr_action).unwrap();
                let next_obs_repr = (self.obs_to_repr)(&obs);
                let next_action_repr: usize = agent.get_action(&next_obs_repr);
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
