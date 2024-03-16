use environments::{classic_control::CartPoleEnv, Env};
use reinforcement_learning::{
    action_selection::epsilon_greedy::{EpsilonDecreasing, EpsilonGreedy, EpsilonUpdateStrategy},
    agent::{ContinuousObsDiscreteActionAgent, DoubleDeepAgent, OptimizerEnum},
    experience_buffer::RandomExperienceBuffer,
};
use std::{collections::VecDeque, rc::Rc};
use tch::{
    nn::{self, Module, VarStore},
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
    const GAMMA: f32 = 0.99;
    const UPDATE_FREQ: i64 = 50;

    let mut state: Tensor;
    let mut action: usize;
    let mut reward: f32;
    let mut done: bool;
    let mut state_: Tensor;

    let device = Device::cuda_if_available();
    let mem_replay = RandomExperienceBuffer::new(1_000, 500, 42, device);

    let epsilon_decreasing = EpsilonDecreasing::new(0.0, Rc::new(move |a| a - 0.001));
    let epsilon_greedy = EpsilonGreedy::new(
        1.0,
        42,
        EpsilonUpdateStrategy::EpsilonDecreasing(epsilon_decreasing),
    );

    let mut agent = DoubleDeepAgent::new(
        Box::new(epsilon_greedy),
        mem_replay,
        generate_policy,
        OptimizerEnum::Adam(nn::Adam::default()),
        0.0005,
        GAMMA,
        device,
    );
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
    };
    for steps in 0..100000 {
        // begin get_action()
        action = agent.get_action(&state) as usize;
        // end get_action()

        (state_, reward, done) = {
            let (state_, reward, done) = env.step(action).unwrap();
            (
                Tensor::from_slice(&[
                    state_.cart_position,
                    state_.cart_velocity,
                    state_.pole_angle,
                    state_.pole_angular_velocity,
                ]),
                reward,
                done,
            )
        };
        // update()
        ep_return += reward;

        agent
            .memory
            .add(&state, action, reward, done, &state_, action);
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
            };

            let avg = ep_returns.average();
            // println!("reward: {}", avg);
            if nepisodes % 100 == 0 {
                println!(
                    "Episode: {}, steps: {}, Avg Return: {} ",
                    nepisodes, steps, avg
                );
            }
            if avg >= 450.0 {
                println!(
                    "Solved at episode {}, step {} with avg of {}",
                    nepisodes, steps, avg,
                );
                break;
            }
            agent.action_selection.update(avg);
        }
        if !agent.memory.ready() {
            continue;
        }

        let (b_state, b_action, b_reward, b_done, b_state_, _) = agent.memory.sample_batch(128);
        let qvalues = agent.policy.forward(&b_state.to_device(device)).gather(
            1,
            &b_action.to_device(device),
            false,
        );

        let max_target_values: Tensor = tch::no_grad(|| {
            agent
                .target_policy
                .forward(&b_state_.to_device(device))
                .max_dim(1, true)
                .0
        });
        let expected_values = b_reward.to_device(device)
            + GAMMA * (&one - &b_done.to_device(device)) * (&max_target_values);
        let expected_values = expected_values.to_kind(Kind::Float).to_device(device);
        let loss = qvalues.mse_loss(&expected_values, tch::Reduction::Mean);
        agent.optimizer.zero_grad();
        loss.backward();
        agent.optimizer.step();
        if nepisodes % UPDATE_FREQ == 0 {
            let a = agent.target_policy_vs.copy(&agent.policy_vs);
            if a.is_err() {
                println!("copy error")
            };
        }
    }
}
