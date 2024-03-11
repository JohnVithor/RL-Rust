use environments::{classic_control::CartPoleEnv, Env};
use rand::{rngs::StdRng, RngCore, SeedableRng};
use reinforcement_learning::{
    action_selection::{ContinuousObsDiscreteActionSelection, EpsilonDecreasing},
    experience_buffer::RandomExperienceBuffer,
};
use std::{collections::VecDeque, rc::Rc, time::Instant};
use tch::{
    nn::{self, Module, Optimizer, OptimizerConfig, VarStore},
    Device, Kind, Tensor,
};
use utils::argmax;

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

pub struct DoubleDeepAgent {
    pub action_selection: EpsilonDecreasing,
    pub discount_factor: f32,
    pub optimizer: Optimizer,
    pub policy: Box<dyn Module>,
    pub target_policy: Box<dyn Module>,
    pub policy_vs: VarStore,
    pub target_policy_vs: VarStore,
    pub memory: RandomExperienceBuffer,
    pub target_update: usize,
    pub device: Device,
}

fn generate_policy(device: Device) -> (Box<dyn Module>, VarStore) {
    const NEURONS: i64 = 128;

    let mem_policy = nn::VarStore::new(device);
    let policy_net = nn::seq()
        .add(nn::linear(
            &mem_policy.root() / "al1",
            4,
            NEURONS,
            Default::default(),
        ))
        .add_fn(|xs| xs.gelu("none"))
        .add(nn::linear(
            &mem_policy.root() / "al2",
            NEURONS,
            2,
            Default::default(),
        ))
        .add_fn(|xs| xs.softmax(0, Kind::Float));
    (Box::new(policy_net), mem_policy)
}

pub fn evaluate(
    env: &mut CartPoleEnv,
    policy_net: &dyn Module,
    n_episodes: u128,
    device: Device,
) -> (Vec<f32>, Vec<u128>) {
    let mut reward_history: Vec<f32> = vec![];
    let mut episode_length: Vec<u128> = vec![];
    for _episode in 0..n_episodes {
        let mut action_counter: u128 = 0;
        let mut epi_reward: f32 = 0.0;
        let state = {
            let s = env.reset();
            Tensor::from_slice(&[
                s.cart_position,
                s.cart_velocity,
                s.pole_angle,
                s.pole_angular_velocity,
            ])
            .to_device(device)
        };
        let values = tch::no_grad(|| policy_net.forward(&state));
        let values: ndarray::ArrayD<f32> = (&values).try_into().unwrap();
        let mut curr_action = argmax(values.iter());
        loop {
            action_counter += 1;
            let (obs, reward, terminated) = env.step(curr_action).unwrap();
            let state = {
                Tensor::from_slice(&[
                    obs.cart_position,
                    obs.cart_velocity,
                    obs.pole_angle,
                    obs.pole_angular_velocity,
                ])
                .to_device(device)
            };
            let values = tch::no_grad(|| policy_net.forward(&state));
            let values: ndarray::ArrayD<f32> = (&values).try_into().unwrap();
            let next_action = argmax(values.iter());
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

fn main() {
    let mut rng: StdRng = StdRng::seed_from_u64(0);
    tch::manual_seed(rng.next_u64() as i64);
    tch::maybe_init_cuda();
    const MEM_SIZE: usize = 5_000;
    const MIN_MEM_SIZE: usize = 1_000;
    const GAMMA: f32 = 0.99;
    const UPDATE_FREQ: i64 = 10;
    const LEARNING_RATE: f64 = 0.0005;
    let device: Device = Device::Cpu;

    let mut state: Tensor;
    let mut action: i64;
    let mut reward: f32;
    let mut done: bool;
    let mut state_: Tensor;

    let mut mem_replay =
        RandomExperienceBuffer::new(MEM_SIZE, MIN_MEM_SIZE, rng.next_u64(), device);

    let (policy_net, mem_policy) = generate_policy(device);
    let (target_net, mut mem_target) = generate_policy(device);
    let mut opt = nn::Adam::default()
        .build(&mem_policy, LEARNING_RATE)
        .unwrap();

    mem_target.copy(&mem_policy).unwrap();

    let mut ep_returns = RunningStat::<f32>::new(50);
    let mut ep_return: f32 = 0.0;
    let mut nepisodes = 0;

    let one: Tensor = Tensor::from(1.0);

    let mut train_env = CartPoleEnv::new(500, rng.next_u64());
    let mut eval_env = CartPoleEnv::new(500, rng.next_u64());
    state = {
        let s = train_env.reset();
        Tensor::from_slice(&[
            s.cart_position,
            s.cart_velocity,
            s.pole_angle,
            s.pole_angular_velocity,
        ])
        .to_device(device)
    };

    let start = Instant::now();

    let mut epsilon_greedy =
        EpsilonDecreasing::new(1.0, Rc::new(|x| x - 0.0005), 0.0, rng.next_u64());

    loop {
        let values = tch::no_grad(|| policy_net.forward(&state));
        let values: ndarray::ArrayD<f32> = (&values).try_into().unwrap();
        let len = values.len();
        action = epsilon_greedy.get_action(&values.into_shape(len).unwrap()) as i64;

        (state_, reward, done) = {
            let (state_, reward, done) = train_env.step(action as usize).unwrap();
            (
                Tensor::from_slice(&[
                    state_.cart_position,
                    state_.cart_velocity,
                    state_.pole_angle,
                    state_.pole_angular_velocity,
                ])
                .to_device(device),
                reward,
                done,
            )
        };
        mem_replay.add(&state, action, reward, done, &state_, action);
        ep_return += reward;
        state = state_;

        if done {
            if nepisodes % 50 == 0 {
                let (r, _l) = evaluate(&mut eval_env, policy_net.as_ref(), 100, device);
                let avg = (r.iter().cloned().sum::<f32>()) / (r.len() as f32);
                println!(
                    "Episode: {}, Avg Return: {:.3} Epsilon: {:.3}",
                    nepisodes,
                    avg,
                    epsilon_greedy.get_epsilon()
                );
                if avg == 500.0 {
                    println!("Solved at episode {} with avg of {}", nepisodes, avg);
                    break;
                }
            }
            nepisodes += 1;
            ep_returns.add(ep_return);
            ep_return = 0.0;
            state = {
                let s = train_env.reset();
                Tensor::from_slice(&[
                    s.cart_position,
                    s.cart_velocity,
                    s.pole_angle,
                    s.pole_angular_velocity,
                ])
                .to_device(device)
            };
            epsilon_greedy.update(0.0);
        }

        if mem_replay.ready() {
            let (b_state, b_action, b_reward, b_done, b_state_, _) = mem_replay.sample_batch(32);
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
    let elapsed = start.elapsed();
    println!("Debug: {:?}", elapsed);
}
