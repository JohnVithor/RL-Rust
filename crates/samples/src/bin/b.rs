use environments::classic_control::{cart_pole::CartPoleObservation, CartPoleEnv};
use fastrand::Rng;
use reinforcement_learning::{
    action_selection::epsilon_greedy::{EpsilonDecreasing, EpsilonGreedy, EpsilonUpdateStrategy},
    agent::{DoubleDeepAgent, OptimizerEnum},
    experience_buffer::RandomExperienceBuffer,
    trainer::ContinuousObsDiscreteTrainer,
};
use std::{rc::Rc, time::Instant};
use tch::{
    nn::{self, Module, VarStore},
    Device, Kind, Tensor,
};

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

fn main() {
    // Lucky: 4,
    // Unlucky: 6
    let mut rng: Rng = Rng::with_seed(42);

    tch::manual_seed(rng.i64(..));
    tch::maybe_init_cuda();
    const MEM_SIZE: usize = 5_000;
    const MIN_MEM_SIZE: usize = 1_000;
    const GAMMA: f32 = 0.9;
    const UPDATE_FREQ: u128 = 10;
    const LEARNING_RATE: f64 = 0.0005;
    const EPSILON_DECAY: f32 = 0.0005;
    const MAX_STEPS_PER_EPI: u128 = 500;
    const START_EPSILON: f32 = 1.0;
    let device: Device = Device::Cpu;

    let mut train_env = CartPoleEnv::new(MAX_STEPS_PER_EPI, rng.u64(..));
    // let mut eval_env = CartPoleEnv::new(MAX_STEPS_PER_EPI, rng.u64(..));

    let mem_replay = RandomExperienceBuffer::new(MEM_SIZE, MIN_MEM_SIZE, rng.u64(..), device);

    let epsilon_decreasing = EpsilonDecreasing::new(0.0, Rc::new(move |a| a - EPSILON_DECAY));
    let epsilon_greedy = EpsilonGreedy::new(
        START_EPSILON,
        rng.u64(..),
        EpsilonUpdateStrategy::EpsilonDecreasing(epsilon_decreasing),
    );

    let mut agent = DoubleDeepAgent::new(
        Box::new(epsilon_greedy),
        mem_replay,
        generate_policy,
        OptimizerEnum::Adam(nn::Adam::default()),
        LEARNING_RATE,
        GAMMA,
        device,
    );

    let mut trainer = ContinuousObsDiscreteTrainer::new(
        |repr: usize| -> usize { repr },
        |repr: &CartPoleObservation| -> Tensor {
            Tensor::from_slice(&[
                repr.cart_position,
                repr.cart_velocity,
                repr.pole_angle,
                repr.pole_angular_velocity,
            ])
        },
    );

    let start = Instant::now();

    trainer.train_by_steps(
        &mut train_env,
        &mut agent,
        100_000,
        UPDATE_FREQ,
        50,
        100,
        false,
    );

    let elapsed = start.elapsed();
    println!("Debug: {:?}", elapsed);
}
