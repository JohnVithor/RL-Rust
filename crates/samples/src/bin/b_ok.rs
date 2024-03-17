use environments::{
    classic_control::{cart_pole::CartPoleObservation, CartPoleEnv},
    Env,
};
use rand::{rngs::StdRng, RngCore, SeedableRng};
use reinforcement_learning::{
    action_selection::epsilon_greedy::{EpsilonGreedy, EpsilonUpdateStrategy},
    agent::{ContinuousObsDiscreteActionAgent, DoubleDeepAgent, OptimizerEnum},
    experience_buffer::RandomExperienceBuffer,
    trainer::ContinuousObsDiscreteTrainer,
};
use std::time::Instant;
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
    let mut rng: StdRng = StdRng::seed_from_u64(4);

    tch::manual_seed(rng.next_u64() as i64);
    tch::maybe_init_cuda();
    const MEM_SIZE: usize = 5_000;
    const MIN_MEM_SIZE: usize = 1_000;
    const GAMMA: f32 = 0.99;
    const UPDATE_FREQ: i64 = 10;
    const LEARNING_RATE: f64 = 0.0005;
    const EPSILON_DECAY: f32 = 0.0005;
    const MAX_STEPS_PER_EPI: u128 = 500;
    const START_EPSILON: f32 = 1.0;
    let device: Device = Device::Cpu;

    let mut _train_env = CartPoleEnv::new(MAX_STEPS_PER_EPI, rng.next_u64());
    let mut _eval_env = CartPoleEnv::new(MAX_STEPS_PER_EPI, rng.next_u64());

    let mut train_env = CartPoleEnv::new(MAX_STEPS_PER_EPI, rng.next_u64());

    let mem_replay = RandomExperienceBuffer::new(MEM_SIZE, MIN_MEM_SIZE, rng.next_u64(), device);

    let epsilon_greedy = EpsilonGreedy::new(
        START_EPSILON,
        rng.next_u64(),
        EpsilonUpdateStrategy::EpsilonDecreasing {
            final_epsilon: 0.0,
            epsilon_decay: Box::new(move |a| a - EPSILON_DECAY),
        },
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

    let mut nepisodes = 0;
    let mut trainer = ContinuousObsDiscreteTrainer::new(
        Box::new(_train_env),
        Box::new(_eval_env),
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
    let mut curr_obs: Tensor = (trainer.obs_to_repr)(&train_env.reset());

    let start = Instant::now();

    loop {
        let curr_action = agent.get_action(&curr_obs);

        let (next_obs, reward, terminated) = train_env
            .step((trainer.repr_to_action)(curr_action))
            .unwrap();
        let next_obs = (trainer.obs_to_repr)(&next_obs);
        agent.add_transition(
            &curr_obs,
            curr_action,
            reward,
            terminated,
            &next_obs,
            curr_action, // HERE
        );

        curr_obs = next_obs;

        agent.update();

        if terminated {
            if nepisodes % UPDATE_FREQ == 0 && agent.update_networks().is_err() {
                println!("copy error")
            }

            if nepisodes % 50 == 0 {
                // let (rewards, eval_lengths) = evaluate(&mut eval_env, &agent, 10, device);
                let (rewards, eval_lengths) = trainer.evaluate(&mut agent, 10);
                let reward_avg = (rewards.iter().sum::<f32>()) / (rewards.len() as f32);
                let _eval_lengths_avg = (eval_lengths.iter().map(|x| *x as f32).sum::<f32>())
                    / (eval_lengths.len() as f32);
                println!("Episode: {}, Avg Return: {:.3} ", nepisodes, reward_avg,);
                if reward_avg == MAX_STEPS_PER_EPI as f32 {
                    println!("Solved at episode {} with avg of {}", nepisodes, reward_avg);
                    break;
                }
            }
            nepisodes += 1;
            curr_obs = (trainer.obs_to_repr)(&train_env.reset());
            agent.action_selection_update(0.0);
        }
    }
    let elapsed = start.elapsed();
    println!("Debug: {:?}", elapsed);
}
