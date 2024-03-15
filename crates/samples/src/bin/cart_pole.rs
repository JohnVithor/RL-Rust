use std::{collections::VecDeque, rc::Rc};

use environments::{classic_control::CartPoleEnv, Env};
use ndarray::Array1;
use reinforcement_learning::{
    action_selection::epsilon_greedy::{
        AdaptativeEpsilon, EpsilonDecreasing, EpsilonGreedy, EpsilonUpdateStrategy,
    },
    agent::{ContinuousObsDiscreteActionAgent, DoubleDeepAgent},
};
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
    let seed: u64 = 42;
    tch::manual_seed(seed as i64);
    tch::maybe_init_cuda();
    const MEM_SIZE: usize = 100_000;
    const MIN_MEM_SIZE: usize = 1_000;
    const GAMMA: f32 = 0.99;
    const UPDATE_FREQ: usize = 10;
    const LEARNING_RATE: f64 = 2.3e-3;

    let device = Device::cuda_if_available();

    let epsilon_decreasing = AdaptativeEpsilon::new(0.0, 450.0, 0.0, 0.5);

    let mut agent = DoubleDeepAgent::new(
        Box::new(EpsilonGreedy::new(
            1.0,
            seed,
            EpsilonUpdateStrategy::AdaptativeEpsilon(epsilon_decreasing),
        )),
        // qlearning,
        LEARNING_RATE,
        GAMMA,
        UPDATE_FREQ,
        generate_policy,
        device,
        Adam::default(),
        MEM_SIZE,
        MIN_MEM_SIZE,
        seed,
    );

    let mut env = CartPoleEnv::default();

    let mut ep_tds = RunningStat::<f32>::new(50);
    let mut ep_td: f32 = 0.0;

    for epi in 1..10_000 {
        let state = env.reset();
        let mut state = Array1::from_vec(vec![
            state.cart_position,
            state.cart_velocity,
            state.pole_angle,
            state.pole_angular_velocity,
        ]);
        let mut action = agent.get_action(&state) as usize;
        loop {
            let (state_, reward, done) = {
                let (state_, reward, done) = env.step(action).unwrap();
                (
                    Array1::from_vec(vec![
                        state_.cart_position,
                        state_.cart_velocity,
                        state_.pole_angle,
                        state_.pole_angular_velocity,
                    ]),
                    reward,
                    done,
                )
            };
            let next_action_repr = agent.get_action(&state) as usize;
            let td = agent.update(&state, action, reward, done, &state_, next_action_repr);
            ep_td += td;
            state = state_;
            action = next_action_repr;
            if done {
                ep_tds.add(ep_td);
                ep_td = 0.0;
                let td_avg = ep_tds.average();
                if epi % 100 == 0 {
                    let mut ep_returns = RunningStat::<f32>::new(50);

                    for _episode in 0..10 {
                        let mut epi_reward: f32 = 0.0;
                        let state = env.reset();
                        let state = Array1::from_vec(vec![
                            state.cart_position,
                            state.cart_velocity,
                            state.pole_angle,
                            state.pole_angular_velocity,
                        ]);
                        let mut curr_action = agent.get_action(&state) as usize;
                        loop {
                            let (state_, reward, done) = {
                                let (state_, reward, done) = env.step(curr_action).unwrap();
                                (
                                    Array1::from_vec(vec![
                                        state_.cart_position,
                                        state_.cart_velocity,
                                        state_.pole_angle,
                                        state_.pole_angular_velocity,
                                    ]),
                                    reward,
                                    done,
                                )
                            };
                            let next_action: usize = agent.get_best_action(&state_);
                            curr_action = next_action;
                            epi_reward += reward;
                            if done {
                                ep_returns.add(epi_reward);
                                break;
                            }
                        }
                    }
                    let r_avg = ep_returns.average();
                    println!(
                        "Episode: {}, Avg Return: {}, Avg TD: {} ",
                        epi, r_avg, td_avg
                    );
                    if r_avg >= 475.0 {
                        println!(
                            "Solved at episode {}, with avg return of {} and avg td of {}",
                            epi, r_avg, td_avg
                        );
                        return;
                    }
                }

                break;
            }
        }
    }
}
