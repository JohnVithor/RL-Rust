use environments::{classic_control::CartPoleEnv, Env};
use rand::{rngs::StdRng, RngCore, SeedableRng};
use reinforcement_learning::{
    action_selection::{
        epsilon_greedy::{EpsilonGreedy, EpsilonUpdateStrategy},
        ContinuousObsDiscreteActionSelection,
    },
    experience_buffer::RandomExperienceBuffer,
};
use std::time::Instant;
use tch::{
    nn::{self, Adam, AdamW, Module, Optimizer, OptimizerConfig, RmsProp, Sgd, VarStore},
    COptimizer, Device, Kind, TchError, Tensor,
};

fn evaluate(
    env: &mut CartPoleEnv,
    agent: &Agent,
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
        let mut curr_action = agent.get_best_action(&state);
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
            curr_action = agent.get_best_action(&state);
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
        // .add_fn(|xs| xs.tanh())
        .add(nn::linear(
            &mem_policy.root() / "al2",
            NEURONS,
            2,
            Default::default(),
        ))
        .add_fn(|xs| xs.softmax(0, Kind::Float));
    (Box::new(policy_net), mem_policy)
}

enum OptimizerEnum {
    Adam(Adam),
    _Sgd(Sgd),
    _RmsProp(RmsProp),
    _AdamW(AdamW),
}

impl OptimizerConfig for OptimizerEnum {
    fn build_copt(&self, lr: f64) -> Result<COptimizer, TchError> {
        match self {
            OptimizerEnum::Adam(opt) => opt.build_copt(lr),
            OptimizerEnum::_Sgd(opt) => opt.build_copt(lr),
            OptimizerEnum::_RmsProp(opt) => opt.build_copt(lr),
            OptimizerEnum::_AdamW(opt) => opt.build_copt(lr),
        }
    }
}

struct Agent {
    action_selection: EpsilonGreedy,
    memory: RandomExperienceBuffer,
    policy: Box<dyn Module>,
    policy_vs: VarStore,
    target_policy: Box<dyn Module>,
    target_policy_vs: VarStore,
    optimizer: Optimizer,
    discount_factor: f32,
}

impl Agent {
    pub fn new(
        epsilon_greedy: EpsilonGreedy,
        mem_replay: RandomExperienceBuffer,
        generate_policy: fn(device: Device) -> (Box<dyn Module>, VarStore),
        opt: OptimizerEnum,
        lr: f64,
        discount_factor: f32,
        device: Device,
    ) -> Self {
        let (policy_net, mem_policy) = generate_policy(device);
        let (target_net, mut mem_target) = generate_policy(device);
        mem_target.copy(&mem_policy).unwrap();
        Self {
            optimizer: opt.build(&mem_policy, lr).unwrap(),
            action_selection: epsilon_greedy,
            memory: mem_replay,
            policy: policy_net,
            policy_vs: mem_policy,
            target_policy: target_net,
            target_policy_vs: mem_target,
            discount_factor,
        }
    }

    fn get_action(&mut self, state: &Tensor) -> usize {
        let values = tch::no_grad(|| self.policy.forward(state));
        let values: ndarray::ArrayD<f32> = (&values).try_into().unwrap();
        let len = values.len();
        self.action_selection
            .get_action(&values.into_shape(len).unwrap()) as usize
    }

    fn get_best_action(&self, state: &Tensor) -> usize {
        let values = tch::no_grad(|| self.policy.forward(state));
        let a: i32 = values.argmax(0, true).try_into().unwrap();
        a as usize
    }

    fn add_transition(
        &mut self,
        curr_state: &Tensor,
        curr_action: usize,
        reward: f32,
        done: bool,
        next_state: &Tensor,
        next_action: usize,
    ) {
        self.memory.add(
            curr_state,
            curr_action,
            reward,
            done,
            next_state,
            next_action,
        );
    }

    fn update_networks(&mut self) -> Result<(), TchError> {
        self.target_policy_vs.copy(&self.policy_vs)
    }

    fn get_batch(&mut self, size: usize) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) {
        self.memory.sample_batch(size)
    }

    fn batch_qvalues(&self, b_states: &Tensor, b_actions: &Tensor) -> Tensor {
        self.policy.forward(b_states).gather(1, b_actions, false)
    }

    fn batch_expected_values(
        &self,
        b_state_: &Tensor,
        b_reward: &Tensor,
        b_done: &Tensor,
    ) -> Tensor {
        let best_target_qvalues =
            tch::no_grad(|| self.target_policy.forward(b_state_).max_dim(1, true).0);
        b_reward + self.discount_factor * (&Tensor::from(1.0) - b_done) * (&best_target_qvalues)
    }

    fn optimize(&mut self, loss: &Tensor) {
        self.optimizer.zero_grad();
        loss.backward();
        self.optimizer.step();
    }
}

fn main() {
    // Lucky: 4,
    // Unlucky: 6
    let mut rng: StdRng = StdRng::seed_from_u64(4);

    tch::manual_seed(rng.next_u64() as i64);
    tch::maybe_init_cuda();
    const MEM_SIZE: usize = 5_000;
    const MIN_MEM_SIZE: usize = 1_000;
    const GAMMA: f32 = 0.9;
    const UPDATE_FREQ: i64 = 10;
    const LEARNING_RATE: f64 = 0.0005;
    const EPSILON_DECAY: f32 = 0.0005;
    const MAX_STEPS_PER_EPI: u128 = 500;
    const START_EPSILON: f32 = 1.0;
    let device: Device = Device::Cpu;

    let mut train_env = CartPoleEnv::new(MAX_STEPS_PER_EPI, rng.next_u64());
    let mut eval_env = CartPoleEnv::new(MAX_STEPS_PER_EPI, rng.next_u64());

    let mem_replay = RandomExperienceBuffer::new(MEM_SIZE, MIN_MEM_SIZE, rng.next_u64(), device);

    let epsilon_greedy = EpsilonGreedy::new(
        START_EPSILON,
        rng.next_u64(),
        EpsilonUpdateStrategy::EpsilonDecreasing {
            final_epsilon: 0.0,
            epsilon_decay: Box::new(move |a| a - EPSILON_DECAY),
        },
    );

    let mut agent = Agent::new(
        epsilon_greedy,
        mem_replay,
        generate_policy,
        OptimizerEnum::Adam(nn::Adam::default()),
        LEARNING_RATE,
        GAMMA,
        device,
    );

    let mut nepisodes = 0;

    let mut state: Tensor;
    let mut action: usize;
    let mut reward: f32;
    let mut done: bool;
    let mut state_: Tensor;

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

    loop {
        action = agent.get_action(&state);

        (state_, reward, done) = {
            let (state_, reward, done) = train_env.step(action).unwrap();
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
        agent.add_transition(&state, action, reward, done, &state_, action);

        state = state_;

        if agent.memory.ready() {
            let (b_state, b_action, b_reward, b_done, b_state_, _) = agent.get_batch(32);
            let policy_qvalues = agent.batch_qvalues(&b_state, &b_action);
            let expected_values = agent.batch_expected_values(&b_state_, &b_reward, &b_done);
            let loss = policy_qvalues.mse_loss(&expected_values, tch::Reduction::Mean);
            agent.optimize(&loss);
        }

        if done {
            if nepisodes % UPDATE_FREQ == 0 && agent.update_networks().is_err() {
                println!("copy error")
            }
            if nepisodes % 50 == 0 {
                let (r, _l) = evaluate(&mut eval_env, &agent, 10, device);
                let avg = (r.iter().sum::<f32>()) / (r.len() as f32);
                println!(
                    "Episode: {}, Avg Return: {:.3} Epsilon: {:.3}",
                    nepisodes,
                    avg,
                    agent.action_selection.get_epsilon()
                );
                if avg == MAX_STEPS_PER_EPI as f32 {
                    println!("Solved at episode {} with avg of {}", nepisodes, avg);
                    break;
                }
            }
            nepisodes += 1;
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
            agent.action_selection.update(0.0);
        }
    }
    let elapsed = start.elapsed();
    println!("Debug: {:?}", elapsed);
}
