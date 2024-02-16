use environments::{classic_control::CartPoleEnv, Env};
/* Deep Deterministic Policy Gradient.

   Continuous control with deep reinforcement learning, Lillicrap et al. 2015
   https://arxiv.org/abs/1509.02971

   See https://spinningup.openai.com/en/latest/algorithms/ddpg.html for a
   reference python implementation.
*/
use tch::{
    kind::{FLOAT_CPU, INT64_CPU},
    nn,
    nn::OptimizerConfig,
    Device,
    Kind::Float,
    Tensor,
};

// The impact of the q value of the next state on the current state's q value.
const GAMMA: f32 = 0.99;
// The weight for updating the target networks.
const TAU: f32 = 0.005;
// The capacity of the replay buffer used for sampling training data.
const REPLAY_BUFFER_CAPACITY: usize = 100_000;
// The training batch size for each training iteration.
const TRAINING_BATCH_SIZE: usize = 100;
// The total number of episodes.
const MAX_EPISODES: usize = 100;
// The maximum length of an episode.
const EPISODE_LENGTH: usize = 200;
// The number of training iterations after one episode finishes.
const TRAINING_ITERATIONS: usize = 200;

// Ornstein-Uhlenbeck process parameters.
const MU: f32 = 0.0;
const THETA: f32 = 0.15;
const SIGMA: f32 = 0.1;

const ACTOR_LEARNING_RATE: f32 = 1e-4;
const CRITIC_LEARNING_RATE: f32 = 1e-3;

struct OuNoise {
    mu: f32,
    theta: f32,
    sigma: f32,
    state: Tensor,
}

impl OuNoise {
    fn new(mu: f32, theta: f32, sigma: f32, num_actions: usize) -> Self {
        let state = Tensor::ones([num_actions as _], FLOAT_CPU);
        Self {
            mu,
            theta,
            sigma,
            state,
        }
    }

    fn sample(&mut self) -> &Tensor {
        let dx = self.theta * (self.mu - &self.state)
            + self.sigma * Tensor::randn(self.state.size(), FLOAT_CPU);
        self.state += dx;
        &self.state
    }
}

struct ReplayBuffer {
    obs: Tensor,
    next_obs: Tensor,
    rewards: Tensor,
    actions: Tensor,
    capacity: usize,
    len: usize,
    i: usize,
}

impl ReplayBuffer {
    fn new(capacity: usize, num_obs: usize, num_actions: usize) -> Self {
        Self {
            obs: Tensor::zeros([capacity as _, num_obs as _], FLOAT_CPU),
            next_obs: Tensor::zeros([capacity as _, num_obs as _], FLOAT_CPU),
            rewards: Tensor::zeros([capacity as _, 1], FLOAT_CPU),
            actions: Tensor::zeros([capacity as _, num_actions as _], FLOAT_CPU),
            capacity,
            len: 0,
            i: 0,
        }
    }

    fn push(&mut self, obs: &Tensor, actions: u8, reward: f32, next_obs: &Tensor) {
        let i = self.i % self.capacity;
        self.obs.get(i as _).copy_(obs);
        self.rewards.get(i as _).copy_(&Tensor::from(reward));
        self.actions.get(i as _).copy_(&Tensor::from(actions));
        self.next_obs.get(i as _).copy_(next_obs);
        self.i += 1;
        if self.len < self.capacity {
            self.len += 1;
        }
    }

    fn random_batch(&self, batch_size: usize) -> Option<(Tensor, Tensor, Tensor, Tensor)> {
        if self.len < 3 {
            return None;
        }

        let batch_size = batch_size.min(self.len - 1);
        let batch_indexes = Tensor::randint((self.len - 2) as _, [batch_size as _], INT64_CPU);

        let states = self.obs.index_select(0, &batch_indexes);
        let next_states = self.next_obs.index_select(0, &batch_indexes);
        let actions = self.actions.index_select(0, &batch_indexes);
        let rewards = self.rewards.index_select(0, &batch_indexes);

        Some((states, actions, rewards, next_states))
    }
}

struct Actor {
    var_store: nn::VarStore,
    network: nn::Sequential,
    device: Device,
    num_obs: usize,
    num_actions: usize,
    opt: nn::Optimizer,
    learning_rate: f32,
}

impl Clone for Actor {
    fn clone(&self) -> Self {
        let mut new = Self::new(self.num_obs, self.num_actions, self.learning_rate);
        new.var_store.copy(&self.var_store).unwrap();
        new
    }
}

impl Actor {
    fn new(num_obs: usize, num_actions: usize, learning_rate: f32) -> Self {
        let var_store = nn::VarStore::new(tch::Device::Cpu);
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

struct Critic {
    var_store: nn::VarStore,
    network: nn::Sequential,
    device: Device,
    num_obs: usize,
    num_actions: usize,
    opt: nn::Optimizer,
    learning_rate: f32,
}

impl Clone for Critic {
    fn clone(&self) -> Self {
        let mut new = Self::new(self.num_obs, self.num_actions, self.learning_rate);
        new.var_store.copy(&self.var_store).unwrap();
        new
    }
}

impl Critic {
    fn new(num_obs: usize, num_actions: usize, learning_rate: f32) -> Self {
        let var_store = nn::VarStore::new(tch::Device::Cpu);
        let opt = nn::Adam::default().build(&var_store, 1e-3).unwrap();
        let p = &var_store.root();
        Self {
            network: nn::seq()
                .add(nn::linear(
                    p / "cl1",
                    (num_obs + num_actions) as _,
                    400,
                    Default::default(),
                ))
                .add_fn(|xs| xs.relu())
                .add(nn::linear(p / "cl2", 400, 300, Default::default()))
                .add_fn(|xs| xs.relu())
                .add(nn::linear(p / "cl3", 300, 1, Default::default())),
            device: p.device(),
            var_store,
            num_obs,
            num_actions,
            opt,
            learning_rate,
        }
    }

    fn forward(&self, obs: &Tensor, actions: &Tensor) -> Tensor {
        let xs = Tensor::cat(&[actions.copy(), obs.copy()], 1);
        xs.to_device(self.device).apply(&self.network)
    }
}

fn track(dest: &mut nn::VarStore, src: &nn::VarStore, tau: f32) {
    tch::no_grad(|| {
        for (dest, src) in dest
            .trainable_variables()
            .iter_mut()
            .zip(src.trainable_variables().iter())
        {
            dest.copy_(&(tau * src + (1.0 - tau) * &*dest));
        }
    })
}

struct Agent {
    actor: Actor,
    actor_target: Actor,

    critic: Critic,
    critic_target: Critic,

    replay_buffer: ReplayBuffer,

    ou_noise: OuNoise,

    train: bool,

    gamma: f32,
    tau: f32,
}

impl Agent {
    fn new(
        actor: Actor,
        critic: Critic,
        ou_noise: OuNoise,
        replay_buffer_capacity: usize,
        train: bool,
        gamma: f32,
        tau: f32,
    ) -> Self {
        let actor_target = actor.clone();
        let critic_target = critic.clone();
        let replay_buffer =
            ReplayBuffer::new(replay_buffer_capacity, actor.num_obs, actor.num_actions);
        Self {
            actor,
            actor_target,
            critic,
            critic_target,
            replay_buffer,
            ou_noise,
            train,
            gamma,
            tau,
        }
    }

    fn actions(&mut self, obs: &Tensor) -> Tensor {
        let mut actions = tch::no_grad(|| self.actor.forward(obs));
        if self.train {
            actions += self.ou_noise.sample();
        }
        actions
    }

    fn remember(&mut self, obs: &Tensor, actions: u8, reward: f32, next_obs: &Tensor) {
        self.replay_buffer.push(obs, actions, reward, next_obs);
    }

    fn train(&mut self, batch_size: usize) {
        let (states, actions, rewards, next_states) =
            match self.replay_buffer.random_batch(batch_size) {
                Some(v) => v,
                _ => return, // We don't have enough samples for training yet.
            };

        let mut q_target = self
            .critic_target
            .forward(&next_states, &self.actor_target.forward(&next_states));
        q_target = rewards + (self.gamma * q_target).detach();

        let q = self.critic.forward(&states, &actions);

        let diff = q_target - q;
        let critic_loss = (&diff * &diff).mean(Float);

        self.critic.opt.zero_grad();
        critic_loss.backward();
        self.critic.opt.step();

        let actor_loss = -self
            .critic
            .forward(&states, &self.actor.forward(&states))
            .mean(Float);

        self.actor.opt.zero_grad();
        actor_loss.backward();
        self.actor.opt.step();

        track(
            &mut self.critic_target.var_store,
            &self.critic.var_store,
            self.tau,
        );
        track(
            &mut self.actor_target.var_store,
            &self.actor.var_store,
            self.tau,
        );
    }
}

pub fn run() {
    let mut env = CartPoleEnv::default();
    println!(
        "action space: {}",
        env.action_space().get_discrete_combinations()
    );
    println!(
        "observation space: {:?}",
        env.observation_space().data.len()
    );

    let num_obs = env.observation_space().data.len();
    let num_actions = env.action_space().data.len();

    let actor = Actor::new(num_obs, num_actions, ACTOR_LEARNING_RATE);
    let critic: Critic = Critic::new(num_obs, num_actions, CRITIC_LEARNING_RATE);
    let ou_noise = OuNoise::new(MU, THETA, SIGMA, num_actions);
    let mut agent = Agent::new(
        actor,
        critic,
        ou_noise,
        REPLAY_BUFFER_CAPACITY,
        true,
        GAMMA,
        TAU,
    );

    for episode in 0..MAX_EPISODES {
        let mut obs = env.reset();

        let mut total_reward = 0.0;
        for _ in 0..EPISODE_LENGTH {
            let v_obs = Tensor::from_slice(&[
                obs.cart_position,
                obs.cart_velocity,
                obs.pole_angle,
                obs.pole_angular_velocity,
            ]);
            let action = u8::try_from(agent.actions(&v_obs)).unwrap();

            let (n_obs, reward, terminated) = env.step(action as usize).unwrap();
            total_reward += reward;

            let nv_obs = Tensor::from_slice(&[
                n_obs.cart_position,
                n_obs.cart_velocity,
                n_obs.pole_angle,
                n_obs.pole_angular_velocity,
            ]);

            agent.remember(&v_obs, action, reward, &nv_obs);

            if terminated {
                break;
            }
            obs = n_obs;
        }

        println!("episode {episode} with total reward of {total_reward}");

        for _ in 0..TRAINING_ITERATIONS {
            agent.train(TRAINING_BATCH_SIZE);
        }
    }
}

fn main() {
    run();
}
