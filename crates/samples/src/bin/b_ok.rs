use environments::{classic_control::CartPoleEnv, Env};
use fastrand::Rng;
use std::collections::VecDeque;
use tch::{
    nn::{self, Module, OptimizerConfig},
    Device, Tensor,
};

const DEVICE: Device = Device::Cpu;

pub fn epsilon_greedy(policy: &nn::Sequential, epsilon: f32, obs: &Tensor, rng: &mut Rng) -> i64 {
    let random_number: f32 = rng.f32();
    if random_number > epsilon {
        let value = tch::no_grad(|| policy.forward(obs));
        value.argmax(0, false).int64_value(&[])
    } else {
        rng.i64(0..2)
    }
}

pub fn epsilon_update(
    cur_reward: f32,
    min_reward: f32,
    max_reward: f32,
    min_eps: f32,
    max_eps: f32,
) -> f32 {
    if cur_reward < min_reward {
        return max_eps;
    }
    let reward_range = max_reward - min_reward;
    let eps_range = max_eps - min_eps;
    let min_update = eps_range / reward_range;
    let new_eps = (max_reward - cur_reward) * min_update;
    if new_eps < min_eps {
        min_eps
    } else {
        new_eps
    }
}

pub struct Transition {
    state: Tensor,
    action: i64,
    reward: f32,
    done: Tensor,
    state_: Tensor,
}

impl Transition {
    pub fn new(state: &Tensor, action: i64, reward: f32, done: bool, state_: &Tensor) -> Self {
        Self {
            state: state.shallow_clone(),
            action,
            reward,
            done: Tensor::from(done as i32 as f32).to_device(DEVICE),
            state_: state_.shallow_clone(),
        }
    }
}

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

pub struct ReplayMemory {
    transitions: VecDeque<Transition>,
    capacity: usize,
    minsize: usize,
}

impl ReplayMemory {
    pub fn new(capacity: usize, minsize: usize) -> Self {
        Self {
            transitions: VecDeque::new(),
            capacity,
            minsize,
        }
    }

    pub fn add(&mut self, transition: Transition) {
        self.transitions.push_back(transition);
        if self.transitions.len() > self.capacity {
            self.transitions.pop_front();
        }
    }

    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }

    pub fn sample_batch(
        &self,
        size: usize,
        rng: &mut Rng,
    ) -> (Tensor, Tensor, Tensor, Tensor, Tensor) {
        let index: Vec<usize> = (0..size)
            .map(|_| rng.usize(0..self.transitions.len()))
            .collect();
        let mut states: Vec<Tensor> = Vec::new();
        let mut actions: Vec<i64> = Vec::new();
        let mut rewards: Vec<f32> = Vec::new();
        let mut dones: Vec<Tensor> = Vec::new();
        let mut states_: Vec<Tensor> = Vec::new();
        index.iter().for_each(|i| {
            let transition = self.transitions.get(*i).unwrap();
            states.push(transition.state.shallow_clone());
            actions.push(transition.action);
            rewards.push(transition.reward);
            dones.push(transition.done.shallow_clone());
            states_.push(transition.state_.shallow_clone());
        });
        (
            Tensor::stack(&states, 0).to_device(DEVICE),
            Tensor::from_slice(actions.as_slice())
                .unsqueeze(1)
                .to_device(DEVICE),
            Tensor::from_slice(rewards.as_slice())
                .unsqueeze(1)
                .to_device(DEVICE),
            Tensor::stack(&dones, 0).unsqueeze(1).to_device(DEVICE),
            Tensor::stack(&states_, 0).to_device(DEVICE),
        )
    }

    pub fn init(&mut self, rng: &mut Rng) {
        let mut env = CartPoleEnv::default();
        let mut state = {
            let s = env.reset();
            Tensor::from_slice(&[
                s.cart_position,
                s.cart_velocity,
                s.pole_angle,
                s.pole_angular_velocity,
            ])
            .to_device(DEVICE)
        };
        let stepskip = 4;
        for s in 0..(self.minsize * stepskip) {
            let action = rng.usize(0..2);
            let (state_, reward, done) = {
                let (state_, reward, done) = env.step(action).unwrap();
                (
                    Tensor::from_slice(&[
                        state_.cart_position,
                        state_.cart_velocity,
                        state_.pole_angle,
                        state_.pole_angular_velocity,
                    ])
                    .to_device(DEVICE),
                    reward,
                    done,
                )
            };
            if s % stepskip == 0 {
                let t = Transition::new(&state, action as i64, reward, done, &state_);
                self.add(t);
            }
            if done {
                state = {
                    let s = env.reset();
                    Tensor::from_slice(&[
                        s.cart_position,
                        s.cart_velocity,
                        s.pole_angle,
                        s.pole_angular_velocity,
                    ])
                    .to_device(DEVICE)
                };
            } else {
                state = state_;
            }
        }
    }
}

fn main() {
    tch::manual_seed(42);
    tch::maybe_init_cuda();
    let mut rng: Rng = Rng::with_seed(42);
    const MEM_SIZE: usize = 2000;
    const MIN_MEM_SIZE: usize = 1000;
    const GAMMA: f32 = 0.99;
    const UPDATE_FREQ: i64 = 10;
    const LEARNING_RATE: f64 = 0.00005;
    let mut epsilon: f32 = 1.0;

    let mut state: Tensor;
    let mut action: i64;
    let mut reward: f32;
    let mut done: bool;
    let mut state_: Tensor;

    let mut mem_replay = ReplayMemory::new(MEM_SIZE, MIN_MEM_SIZE);
    let mem_policy = nn::VarStore::new(DEVICE);
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
    let mut opt = nn::Adam::default()
        .build(&mem_policy, LEARNING_RATE)
        .unwrap();
    let mut mem_target = nn::VarStore::new(DEVICE);
    let target_net = nn::seq()
        .add(nn::linear(
            &mem_target.root() / "al1",
            4,
            128,
            Default::default(),
        ))
        .add_fn(|xs| xs.tanh())
        .add(nn::linear(
            &mem_target.root() / "al2",
            128,
            2,
            Default::default(),
        ));
    mem_target.copy(&mem_policy).unwrap();
    let mut ep_returns = RunningStat::<f32>::new(50);
    let mut ep_return: f32 = 0.0;
    let mut nepisodes = 0;
    let one: Tensor = Tensor::from(1.0);

    let mut env = CartPoleEnv::default();
    state = {
        let s = env.reset();
        Tensor::from_slice(&[
            s.cart_position,
            s.cart_velocity,
            s.pole_angle,
            s.pole_angular_velocity,
        ])
        .to_device(DEVICE)
    };
    // mem_replay.init(&mut rng);
    loop {
        action = epsilon_greedy(&policy_net, epsilon, &state, &mut rng);
        (state_, reward, done) = {
            let (state_, reward, done) = env.step(action as usize).unwrap();
            (
                Tensor::from_slice(&[
                    state_.cart_position,
                    state_.cart_velocity,
                    state_.pole_angle,
                    state_.pole_angular_velocity,
                ])
                .to_device(DEVICE),
                reward,
                done,
            )
        };
        let t = Transition::new(&state, action, reward, done, &state_);
        mem_replay.add(t);
        ep_return += reward;
        state = state_;

        if done {
            nepisodes += 1;
            ep_returns.add(ep_return);
            ep_return = 0.0;
            state = {
                let s = env.reset();
                Tensor::from_slice(&[
                    s.cart_position,
                    s.cart_velocity,
                    s.pole_angle,
                    s.pole_angular_velocity,
                ])
                .to_device(DEVICE)
            };

            let avg = ep_returns.average();
            if nepisodes % 100 == 0 {
                println!(
                    "Episode: {}, Avg Return: {} Epsilon: {}",
                    nepisodes, avg, epsilon
                );
            }
            if avg >= 475.0 {
                println!("Solved at episode {} with avg of {}", nepisodes, avg);
                break;
            }
            epsilon = epsilon_update(avg, 0.0, 500.0, 0.05, 0.5);
        }

        if mem_replay.len() >= MIN_MEM_SIZE {
            let (b_state, b_action, b_reward, b_done, b_state_) =
                mem_replay.sample_batch(128, &mut rng);
            let qvalues = policy_net.forward(&b_state).gather(1, &b_action, false);

            let max_target_values: Tensor =
                tch::no_grad(|| target_net.forward(&b_state_).max_dim(1, true).0);
            let expected_values = b_reward + GAMMA * (&one - &b_done) * (&max_target_values);
            let loss = qvalues.mse_loss(&expected_values, tch::Reduction::Mean);
            opt.zero_grad();
            loss.backward();
            opt.step();
            if nepisodes % UPDATE_FREQ == 0 {
                let a = mem_target.copy(&mem_policy);
                if a.is_err() {
                    println!("copy error")
                };
            }
        }
    }
}
