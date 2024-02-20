use std::{
    collections::VecDeque,
    io::{self, BufRead},
};

use environments::{classic_control::CartPoleEnv, Env};
use ndarray::Array1;
use reinforcement_learning::{
    action_selection::AdaptativeEpsilon,
    agent::{ContinuousObsDiscreteActionAgent, DoubleDeepAgent},
};
use samples::Cli;
use structopt::StructOpt;
use tch::{
    nn::{self, Adam, Module, VarStore},
    Device,
};

pub struct RunningStat<T> {
    values: VecDeque<T>,
    capacity: usize,
}

impl<T> RunningStat<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            values: VecDeque::new(),
            capacity,
        }
    }

    pub fn add(&mut self, val: T) {
        self.values.push_back(val);
        if self.values.len() > self.capacity {
            self.values.pop_front();
        }
    }

    pub fn average(&self) -> f32
    where
        T: std::iter::Sum,
        T: std::ops::Div<f32, Output = T>,
        T: Clone,
        T: Into<f32>,
        f32: std::iter::Sum<T>,
    {
        let sum: f32 = self.values.iter().cloned().sum();
        sum / (self.capacity as f32)
    }
}

fn generate_policy(device: Device) -> (Box<dyn Module>, VarStore) {
    let mem_policy = nn::VarStore::new(device);
    let policy_net = nn::seq()
        .add(nn::linear(
            &mem_policy.root() / "al1",
            4,
            128,
            Default::default(),
        ))
        .add_fn(|xs| xs.tanh())
        .add(nn::linear(
            &mem_policy.root() / "al2",
            128,
            2,
            Default::default(),
        ));
    (Box::new(policy_net), mem_policy)
}

fn main() {
    tch::manual_seed(42);
    tch::maybe_init_cuda();
    let debug = false;
    let cli: Cli = Cli::from_args();

    let n_episodes: u128 = cli.n_episodes;
    let max_steps: u128 = cli.max_steps;

    let mut env = CartPoleEnv::new(max_steps, cli.seed);
    let device = Device::Cpu;

    let mut agent = DoubleDeepAgent::new(
        Box::new(AdaptativeEpsilon::new(0.0, 450.0, 0.0, 0.5, 42)),
        // qlearning,
        0.005,
        0.99,
        50,
        generate_policy,
        device,
        Adam::default(),
        2048,
        1024,
        42,
    );

    let eval_at = 100;

    let mut training_reward = RunningStat::<f32>::new(50);

    for episode in 0..n_episodes {
        let mut epi_reward: f32 = 0.0;
        let curr_obs = env.reset();
        let mut curr_obs_repr = Array1::from_vec(vec![
            curr_obs.cart_position,
            curr_obs.cart_velocity,
            curr_obs.pole_angle,
            curr_obs.pole_angular_velocity,
        ]);
        let mut curr_action_repr: usize = agent.get_action(&curr_obs_repr);

        loop {
            let (next_obs, reward, terminated) = env.step(curr_action_repr).unwrap();
            if debug {
                println!("{}", env.render());
                let mut line = String::new();
                let stdin = io::stdin();
                stdin.lock().read_line(&mut line).unwrap();
            }
            let next_obs_repr = Array1::from_vec(vec![
                next_obs.cart_position,
                next_obs.cart_velocity,
                next_obs.pole_angle,
                next_obs.pole_angular_velocity,
            ]);
            let next_action_repr: usize = agent.get_action(&next_obs_repr);
            let _td = agent.update(
                &curr_obs_repr,
                curr_action_repr,
                reward,
                terminated,
                &next_obs_repr,
                next_action_repr,
            );
            curr_obs_repr = next_obs_repr;
            curr_action_repr = next_action_repr;
            epi_reward += reward;
            if terminated {
                if debug {
                    println!("{}", env.render());
                }
                training_reward.add(epi_reward);
                break;
            }
        }
        if episode % eval_at == 0 {
            println!(
                "Episode: {} Mean Rewards: {}",
                episode,
                training_reward.average(),
            );
        }
    }
}
