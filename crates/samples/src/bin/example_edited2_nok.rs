use environments::{classic_control::CartPoleEnv, Env};
use reinforcement_learning::{
    action_selection::{AdaptativeEpsilon, ContinuousObsDiscreteActionSelection},
    experience_buffer::RandomExperienceBuffer,
};
use std::collections::VecDeque;
use tch::{
    nn::{self, Module, OptimizerConfig, VarStore},
    Device, Kind, Tensor,
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
    const MEM_SIZE: usize = 100_000;
    const MIN_MEM_SIZE: usize = 10_000;
    const GAMMA: f32 = 0.99;
    const UPDATE_FREQ: i64 = 50;
    const LEARNING_RATE: f64 = 0.00025;

    let mut action: usize;
    let mut reward: f32;
    let mut done: bool;
    let mut state_: Tensor;

    let device = Device::cuda_if_available();
    let mut mem_replay = RandomExperienceBuffer::new(MEM_SIZE, MIN_MEM_SIZE, 42, device);
    let (policy, policy_vs) = generate_policy(device);
    let (target_policy, mut target_policy_vs) = generate_policy(device);
    target_policy_vs.copy(&policy_vs).unwrap();

    let mut opt = nn::Adam::default()
        .build(&policy_vs, LEARNING_RATE)
        .unwrap();

    let mut eps = AdaptativeEpsilon::new(0.0, 450.0, 0.0, 0.5, 42);

    let mut ep_returns = RunningStat::<f32>::new(50);
    let mut ep_return: f32 = 0.0;
    let one: Tensor = Tensor::from(1.0);

    let mut env = CartPoleEnv::default();

    for epi in 1..10000 {
        let state = env.reset();
        let mut state = Tensor::from_slice(&[
            state.cart_position,
            state.cart_velocity,
            state.pole_angle,
            state.pole_angular_velocity,
        ])
        .to_device(device);
        let values = tch::no_grad(|| policy.forward(&state));
        let values: ndarray::ArrayD<f32> = (&values).try_into().unwrap();
        let values_len = values.len();
        action = eps.get_action(&values.into_shape(values_len).unwrap());
        loop {
            (state_, reward, done) = {
                let (state_, reward, done) = env.step(action).unwrap();
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

            ep_return += reward;

            let values = tch::no_grad(|| policy.forward(&state_));
            let values: ndarray::ArrayD<f32> = (&values).try_into().unwrap();
            let values_len = values.len();
            let n_action = eps.get_action(&values.into_shape(values_len).unwrap());

            mem_replay.add(
                &state,
                action as i64,
                reward,
                done,
                &state_,
                n_action as i64,
            );

            if mem_replay.ready() {
                let (b_state, b_action, b_reward, b_done, b_state_, _) =
                    mem_replay.sample_batch(128);
                let qvalues = policy.forward(&b_state.to_device(device)).gather(
                    1,
                    &b_action.to_device(device),
                    false,
                );
                let max_target_values: Tensor = tch::no_grad(|| {
                    target_policy
                        .forward(&b_state_.to_device(device))
                        .max_dim(1, true)
                        .0
                });
                let expected_values = b_reward.to_device(device)
                    + GAMMA * (&one - &b_done.to_device(device)) * (&max_target_values);
                let expected_values = expected_values.to_kind(Kind::Float).to_device(device);
                let loss: Tensor = qvalues.huber_loss(&expected_values, tch::Reduction::Mean, 1.0);
                opt.zero_grad();
                loss.backward();
                opt.step();
                if done {
                    if epi % UPDATE_FREQ == 0 {
                        let a = target_policy_vs.copy(&policy_vs);
                        if a.is_err() {
                            println!("copy error")
                        };
                    }
                    ep_returns.add(ep_return);
                    ep_return = 0.0;
                    eps.update(ep_returns.average());
                }
            }

            state = state_;
            action = n_action;

            if done {
                if !mem_replay.ready() {
                    ep_returns.add(ep_return);
                    ep_return = 0.0;
                }
                let avg: f32 = ep_returns.average();
                if epi % 100 == 0 {
                    println!(
                        "Episode: {}, Avg Return: {} Epsilon: {}",
                        epi, avg, eps.epsilon
                    );
                }
                if avg >= 450.0 {
                    println!("Solved at episode {}, with avg of {}", epi, avg);
                    return;
                }
                break;
            }
        }
    }
}
