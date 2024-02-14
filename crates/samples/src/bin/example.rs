use environments::{classic_control::CartPoleEnv, Env};
use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use tch::{
    nn::{self, ModuleT, OptimizerConfig},
    Device, Tensor,
};

pub struct ReplayBuffer {
    index: usize,
    pub size: usize,
    capacity: usize,
    states: Array2<f64>,
    actions: Array1<u8>,
    rewards: Array1<f64>,
    next_states: Array2<f64>,
    dones: Array1<bool>,
}

type BufferSample = (
    Array2<f64>,
    Array1<u8>,
    Array1<f64>,
    Array2<f64>,
    Array1<bool>,
);

impl ReplayBuffer {
    pub fn new(capacity: usize, obs_space: usize) -> Self {
        Self {
            index: 0,
            size: 0,
            capacity,
            states: Array2::zeros((capacity, obs_space)),
            actions: Array1::zeros(capacity),
            rewards: Array1::zeros(capacity),
            next_states: Array2::zeros((capacity, obs_space)),
            dones: Array1::default(capacity),
        }
    }
    pub fn update(
        &mut self,
        state: Array1<f64>,
        action: u8,
        reward: f64,
        next_state: Array1<f64>,
        is_terminal: bool,
    ) {
        self.states.row_mut(self.index).assign(&state);
        self.actions[self.index] = action;
        self.rewards[self.index] = reward;
        self.next_states.row_mut(self.index).assign(&next_state);
        self.dones[self.index] = is_terminal;

        self.index = (self.index + 1) % self.capacity;
        if self.size < self.capacity {
            self.size += 1;
        }
    }

    pub fn sample(&self, batch_size: usize) -> BufferSample {
        let mut rng = rand::thread_rng();
        let indexes = rand::seq::index::sample(&mut rng, self.index, batch_size).into_vec();
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
pub struct LinearNetwork {
    l1: nn::Linear,
    l2: nn::Linear,
    l3: nn::Linear,
}

impl LinearNetwork {
    pub fn new(vs: &nn::Path, input_size: i64, output_size: i64) -> Self {
        let l1 = nn::linear(vs, input_size, 128, Default::default());
        let l2 = nn::linear(vs, 128, 128, Default::default());
        let l3 = nn::linear(vs, 128, output_size, Default::default());
        Self { l1, l2, l3 }
    }
}

impl nn::ModuleT for LinearNetwork {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        xs.apply(&self.l1)
            .relu()
            .apply(&self.l2)
            .relu()
            .apply(&self.l3)
    }

    fn batch_accuracy_for_logits(
        &self,
        xs: &tch::Tensor,
        ys: &tch::Tensor,
        d: tch::Device,
        batch_size: i64,
    ) -> f64 {
        let _no_grad = tch::no_grad_guard();
        let mut sum_accuracy = 0f64;
        let mut sample_count = 0f64;
        for (xs, ys) in tch::data::Iter2::new(xs, ys, batch_size).return_smaller_last_batch() {
            let acc = self
                .forward_t(&xs.to_device(d), false)
                .accuracy_for_logits(&ys.to_device(d));
            let size = xs.size()[0] as f64;
            sum_accuracy += f64::try_from(&acc).unwrap() * size;
            sample_count += size;
        }
        sum_accuracy / sample_count
    }
}

pub struct DeepQLearningAgent {
    // learning_rate: f64,
    pub epsilon: f64,
    epsilon_decay: f64,
    final_epsilon: f64,
    discount_factor: f64,
    batch_size: usize,
    // max_memory: i64,
    // device: VarStore,
    network: LinearNetwork,
    replay_buffer: ReplayBuffer,
    optimizer: nn::Optimizer,
    action_space: i64,
}

impl DeepQLearningAgent {
    pub fn new(
        learning_rate: f64,
        epsilon: f64,
        epsilon_decay: f64,
        final_epsilon: f64,
        discount_factor: f64,
        batch_size: usize,
        max_memory: i64,
        obs_space: i64,
        action_space: i64,
    ) -> Self {
        let vs = nn::VarStore::new(Device::cuda_if_available());
        Self {
            // learning_rate,
            epsilon,
            epsilon_decay,
            final_epsilon,
            discount_factor,
            batch_size,
            action_space,
            // max_memory,
            optimizer: nn::Adam::default().build(&vs, learning_rate).unwrap(),
            network: LinearNetwork::new(&vs.root(), obs_space, action_space),
            replay_buffer: ReplayBuffer::new(max_memory as usize, obs_space as usize),
            // device: vs,
        }
    }
    pub fn act(&self, obs: &Tensor) -> usize {
        let mut action = 0;
        tch::no_grad(|| {
            let v = self.network.forward_t(obs, false).argmax(-1, false);
            action = f64::try_from(v).unwrap() as usize;
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
        action: u8,
        reward: f64,
        new_state: Tensor,
        is_terminal: bool,
    ) {
        let nd_state: ndarray::ArrayD<f64> = (&state).try_into().unwrap();
        let nd_n_state: ndarray::ArrayD<f64> = (&new_state).try_into().unwrap();
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

        let states = Tensor::from_slice(states.as_slice().unwrap());
        let actions = Tensor::from_slice(actions.as_slice().unwrap());
        let rewards = Tensor::from_slice(rewards.as_slice().unwrap());
        let next_states = Tensor::from_slice(next_states.as_slice().unwrap());
        let dones = Tensor::from_slice(dones.as_slice().unwrap());

        let q1 = self
            .network
            .forward_t(&states, true)
            .gather(-1, &actions, false);

        let mut target = Tensor::default();
        tch::no_grad(|| {
            let q2 = self
                .network
                .forward_t(&next_states, false)
                .gather(-1, &actions, false)
                .max_dim(-1, true)
                .0;
            target = rewards + (1 - dones) * self.discount_factor * q2;
        });
        let temporal_difference_loss = q1.mse_loss(&target, tch::Reduction::Mean);
        // self.training_error.append(temporal_difference_loss.item())
        self.optimizer.zero_grad();
        temporal_difference_loss.backward();
        self.optimizer.step();
    }
    pub fn decay_epsilon(&mut self) {
        self.epsilon = self.final_epsilon.max(self.epsilon - self.epsilon_decay)
    }
}

pub fn test_accurracy(
    env: &mut CartPoleEnv,
    agent: &mut DeepQLearningAgent,
    num_steps: usize,
    num_episodes: usize,
) -> f64 {
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
            ]);
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
    nb_success / (num_episodes as f64)
}
fn main() {
    tch::manual_seed(42);
    let learning_rate = 1e-4;
    let start_epsilon = 1.0;
    let final_epsilon = 0.1;
    let nb_max_episodes = 400;
    let discount_factor = 0.99;
    let batch_size = 32;
    let max_memory = 10_000;
    let test_freq = 100;

    let mut total_steps = 0;
    let mut env: CartPoleEnv = CartPoleEnv::new(100);
    let mut agent = DeepQLearningAgent::new(
        learning_rate,
        start_epsilon,
        start_epsilon / nb_max_episodes as f64,
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
            ]);
            let action = agent.choose_action(&tensor);
            if let Ok((next_state, reward, t)) = env.step(action) {
                is_terminal = t;
                let n_tensor = Tensor::from_slice(&[
                    next_state.cart_position,
                    next_state.cart_velocity,
                    next_state.pole_angle,
                    next_state.pole_angular_velocity,
                ]);
                agent.remember(tensor, action as u8, reward, n_tensor, is_terminal);
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
            let s: f64 = mean_rewards.iter().sum();
            println!("step: {}, episode: {}, training reward mean: {}, test reward mean: {}, random move probability: {}",total_steps, epi+1, s/test_freq as f64, accur, agent.epsilon);
            mean_rewards.clear();
        }
        rewards.push(total_reward);
        steps.push(episode_step);
    }
    println!(
        "total steps: {}, total episodes: {}",
        total_steps, nb_max_episodes
    );
}
