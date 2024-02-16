use environments::{classic_control::CartPoleEnv, Env};
use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use tch::{
    nn::{self, OptimizerConfig},
    Device, Tensor,
};

pub struct ReplayBuffer {
    index: usize,
    pub size: usize,
    capacity: usize,
    states: Array2<f32>,
    actions: Array1<i64>,
    rewards: Array1<f32>,
    next_states: Array2<f32>,
    dones: Array1<u8>,
}

type BufferSample = (
    Array2<f32>,
    Array1<i64>,
    Array1<f32>,
    Array2<f32>,
    Array1<u8>,
);

impl ReplayBuffer {
    pub fn new(capacity: usize, obs_space: usize) -> Self {
        Self {
            index: 0,
            size: 0,
            capacity,
            states: Array2::zeros((capacity, obs_space)),
            // actions: Array2::zeros((capacity, 1)),
            actions: Array1::zeros(capacity),
            rewards: Array1::zeros(capacity),
            next_states: Array2::zeros((capacity, obs_space)),
            dones: Array1::default(capacity),
        }
    }
    pub fn update(
        &mut self,
        state: Array1<f32>,
        action: i64,
        reward: f32,
        next_state: Array1<f32>,
        is_terminal: bool,
    ) {
        self.states.row_mut(self.index).assign(&state);
        self.actions[self.index] = action;
        self.rewards[self.index] = reward;
        self.next_states.row_mut(self.index).assign(&next_state);
        self.dones[self.index] = is_terminal as u8;

        self.index = (self.index + 1) % self.capacity;
        if self.size < self.capacity {
            self.size += 1;
        }
    }

    pub fn sample(&self, batch_size: usize) -> BufferSample {
        let mut rng = rand::thread_rng();
        // print!("sample {} {} {}", self.index, self.size, batch_size);
        let indexes = rand::seq::index::sample(&mut rng, self.size, batch_size).into_vec();
        (
            self.states.select(Axis(0), &indexes),
            self.actions.select(Axis(0), &indexes),
            self.rewards.select(Axis(0), &indexes),
            self.next_states.select(Axis(0), &indexes),
            self.dones.select(Axis(0), &indexes),
        )
    }
}

#[derive(Debug)]
struct LinearNetwork {
    var_store: nn::VarStore,
    network: nn::Sequential,
    device: Device,
    num_obs: usize,
    num_actions: usize,
    opt: nn::Optimizer,
    learning_rate: f32,
}

impl Clone for LinearNetwork {
    fn clone(&self) -> Self {
        let mut new = Self::new(self.num_obs, self.num_actions, self.learning_rate);
        new.var_store.copy(&self.var_store).unwrap();
        new
    }
}

impl LinearNetwork {
    fn new(num_obs: usize, num_actions: usize, learning_rate: f32) -> Self {
        let var_store = nn::VarStore::new(tch::Device::Cuda(0));
        let opt = nn::Adam::default()
            .build(&var_store, learning_rate.into())
            .unwrap();
        let p = &var_store.root();
        Self {
            network: nn::seq()
                .add(nn::linear(p / "al1", num_obs as _, 400, Default::default()))
                .add_fn(|xs| xs.relu())
                .add(nn::linear(p / "al2", 400, 300, Default::default()))
                .add_fn(|xs| xs.relu())
                .add(nn::linear(
                    p / "al3",
                    300,
                    num_actions as _,
                    Default::default(),
                ))
                .add_fn(|xs| xs.tanh()),
            device: p.device(),
            num_obs,
            num_actions,
            var_store,
            opt,
            learning_rate,
        }
    }

    fn forward(&self, obs: &Tensor) -> Tensor {
        obs.to_device(self.device).apply(&self.network)
    }
}

pub struct DeepQLearningAgent {
    pub epsilon: f32,
    epsilon_decay: f32,
    final_epsilon: f32,
    discount_factor: f32,
    batch_size: usize,
    network: LinearNetwork,
    target_network: LinearNetwork,
    replay_buffer: ReplayBuffer,
    // optimizer: nn::Optimizer,
    action_space: i64,
    pub training_error: Vec<f64>,
}

impl DeepQLearningAgent {
    pub fn new(
        learning_rate: f32,
        epsilon: f32,
        epsilon_decay: f32,
        final_epsilon: f32,
        discount_factor: f32,
        batch_size: usize,
        max_memory: i64,
        obs_space: i64,
        action_space: i64,
    ) -> Self {
        Self {
            epsilon,
            epsilon_decay,
            final_epsilon,
            discount_factor,
            batch_size,
            action_space,
            // optimizer: nn::Adam::default()
            //     .build(&vs_optimizer, learning_rate.into())
            //     .unwrap(),
            network: LinearNetwork::new(obs_space as usize, action_space as usize, learning_rate),
            target_network: LinearNetwork::new(
                obs_space as usize,
                action_space as usize,
                learning_rate,
            ),
            replay_buffer: ReplayBuffer::new(max_memory as usize, obs_space as usize),
            training_error: vec![],
        }
    }
    pub fn act(&self, obs: &Tensor) -> usize {
        let mut action = 0;
        tch::no_grad(|| {
            let v = self.network.forward(obs).argmax(-1, false);
            action = f32::try_from(v).unwrap() as usize;
        });
        action
    }
    pub fn choose_action(&self, obs: &Tensor) -> usize {
        if rand::thread_rng().gen_range(0.0..1.0) < self.epsilon {
            rand::thread_rng().gen_range(0..self.action_space) as usize
        } else {
            self.act(obs)
        }
    }
    pub fn remember(
        &mut self,
        state: Tensor,
        action: i64,
        reward: f32,
        new_state: Tensor,
        is_terminal: bool,
    ) {
        let nd_state: ndarray::ArrayD<f32> = (&state).try_into().unwrap();
        let nd_n_state: ndarray::ArrayD<f32> = (&new_state).try_into().unwrap();
        self.replay_buffer.update(
            nd_state.into_dimensionality().unwrap(),
            action,
            reward,
            nd_n_state.into_dimensionality().unwrap(),
            is_terminal,
        )
    }
    pub fn update(&mut self) {
        if self.batch_size * 10 > self.replay_buffer.size {
            return;
        }
        let (states, actions, rewards, next_states, dones) =
            self.replay_buffer.sample(self.batch_size);

        let states = Tensor::try_from(states).unwrap().to_device(Device::Cuda(0));
        let actions = Tensor::try_from(actions)
            .unwrap()
            .unsqueeze(-1)
            .to_device(Device::Cuda(0));
        let rewards = Tensor::try_from(rewards)
            .unwrap()
            .unsqueeze(-1)
            .to_device(Device::Cuda(0));
        let next_states = Tensor::try_from(next_states)
            .unwrap()
            .to_device(Device::Cuda(0));
        let dones = Tensor::try_from(dones)
            .unwrap()
            .unsqueeze(-1)
            .to_device(Device::Cuda(0));

        let q1 = self.network.forward(&states).gather(-1, &actions, false);

        let q2 = tch::no_grad(|| -> Tensor {
            self.target_network
                .forward(&next_states)
                .gather(-1, &actions, false)
                .max_dim(-1, true)
                .0
        });

        let target = (rewards + (1 - dones)) * (self.discount_factor * q2);
        let temporal_difference_loss = q1.mse_loss(&target, tch::Reduction::Mean);
        // println!("Training error: {:?}", temporal_difference_loss);
        self.training_error
            .push(temporal_difference_loss.double_value(&[]));
        self.network.opt.zero_grad();
        temporal_difference_loss.backward();
        self.network.opt.step();
        self.sync();
    }
    pub fn decay_epsilon(&mut self) {
        self.epsilon = self.final_epsilon.max(self.epsilon - self.epsilon_decay)
    }
    fn sync(&mut self) {
        tch::no_grad(|| {
            // for (dest, src) in self
            //     .network
            //     .var_store
            //     .trainable_variables()
            //     .iter_mut()
            //     .zip(self.target_network.var_store.trainable_variables().iter())
            // {
            //     dest.copy_(&(0.005 * src + (1.0 - 0.005) * &*dest));
            // }
            self.target_network = self.network.clone();
        });
    }
}

pub fn test_accurracy(
    env: &mut CartPoleEnv,
    agent: &mut DeepQLearningAgent,
    num_steps: usize,
    num_episodes: usize,
) -> f32 {
    let mut counter = 0;
    let mut nb_success = 0.0;

    while counter < num_episodes {
        let state = env.reset();

        for _ in 0..num_steps {
            let tensor = Tensor::from_slice(&[
                state.cart_position,
                state.cart_velocity,
                state.pole_angle,
                state.pole_angular_velocity,
            ])
            .to_device(Device::Cuda(0));
            let action = agent.act(&tensor);

            if let Ok((_state, reward, is_terminal)) = env.step(action) {
                nb_success += reward;
                if is_terminal {
                    break;
                }
            } else {
                println!("Error in env.step()");
            }
        }
        counter += 1
    }
    nb_success / (num_episodes as f32)
}
fn main() {
    tch::manual_seed(42);
    tch::maybe_init_cuda();
    let learning_rate = 2.3e-3;
    let start_epsilon = 0.5;
    let final_epsilon = 0.05;
    let nb_max_episodes = 2000;
    let discount_factor = 0.99;
    let batch_size = 64;
    let max_memory = 10_000;
    let test_freq = 100;

    let mut total_steps = 0;
    let mut env: CartPoleEnv = CartPoleEnv::new(100);
    let mut agent = DeepQLearningAgent::new(
        learning_rate,
        start_epsilon,
        start_epsilon / nb_max_episodes as f32,
        final_epsilon,
        discount_factor,
        batch_size,
        max_memory,
        4,
        1,
    );

    let mut accuracies = vec![];
    let mut mean_rewards = vec![];
    let mut rewards = vec![];
    let mut steps = vec![];

    for epi in 0..nb_max_episodes as usize {
        let mut state = env.reset();
        let mut is_terminal = false;
        let mut total_reward = 0.0;
        let mut episode_step: usize = 0;
        while !is_terminal {
            episode_step += 1;
            let tensor = Tensor::from_slice(&[
                state.cart_position,
                state.cart_velocity,
                state.pole_angle,
                state.pole_angular_velocity,
            ])
            .to_device(Device::Cuda(0));
            let action: usize = agent.choose_action(&tensor);
            if let Ok((next_state, reward, t)) = env.step(action) {
                is_terminal = t;
                let n_tensor = Tensor::from_slice(&[
                    next_state.cart_position,
                    next_state.cart_velocity,
                    next_state.pole_angle,
                    next_state.pole_angular_velocity,
                ])
                .to_device(Device::Cuda(0));
                agent.remember(tensor, action as i64, reward, n_tensor, is_terminal);
                agent.update();
                state = next_state;
                total_steps += 1;
                total_reward += reward;
            } else {
                println!("Error in env.step()");
            }
        }
        agent.decay_epsilon();
        mean_rewards.push(total_reward);
        if (epi + 1) % test_freq == 0 {
            let accur = test_accurracy(&mut env, &mut agent, 400, 25);
            accuracies.push(accur);
            let s: f32 = mean_rewards.iter().sum();
            println!("step: {}, episode: {}, training reward mean: {}, test reward mean: {}, random move probability: {}",total_steps, epi+1, s/test_freq as f32, accur, agent.epsilon);
            mean_rewards.clear();
        }
        rewards.push(total_reward);
        steps.push(episode_step);
    }
    println!(
        "total steps: {}, total episodes: {}, ",
        total_steps, nb_max_episodes
    );
    println!(
        "training error mean: {}",
        agent.training_error.iter().sum::<f64>() / nb_max_episodes as f64
    );
}
