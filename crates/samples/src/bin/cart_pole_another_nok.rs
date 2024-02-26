use environments::classic_control::{cart_pole::CartPoleObservation, CartPoleEnv};
use ndarray::Array1;
use reinforcement_learning::{
    action_selection::AdaptativeEpsilon, agent::DoubleDeepAgent,
    trainer::ContinuousObsDiscreteTrainer,
};
use tch::{
    nn::{self, Adam, Module, VarStore},
    Device,
};

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

    let mut agent = DoubleDeepAgent::new(
        Box::new(AdaptativeEpsilon::new(0.0, 450.0, 0.0, 0.5, seed)),
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

    let mut trainer = ContinuousObsDiscreteTrainer::new(
        |repr: usize| -> usize { repr },
        |repr: &CartPoleObservation| -> Array1<f32> {
            Array1::from_vec(vec![
                repr.cart_position,
                repr.cart_velocity,
                repr.pole_angle,
                repr.pole_angular_velocity,
            ])
        },
    );
    let (training_reward, training_length, training_error, _evaluation_reward, _evaluation_length) =
        trainer.train_by_episode(&mut env, &mut agent, 1000, 100, 100, false);

    println!(
        "Training reward: {:?}",
        training_reward.iter().sum::<f32>() / training_reward.len() as f32
    );
    println!(
        "Training length: {:?}",
        training_length.iter().sum::<u128>() / training_length.len() as u128
    );
    println!(
        "Training error: {:?}",
        training_error.iter().sum::<f32>() / training_error.len() as f32
    );
}
