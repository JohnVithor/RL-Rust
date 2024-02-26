use environments::{
    classic_control::{cart_pole::CartPoleObservation, CartPoleEnv},
    Env,
};
use ndarray::{Array, Array1};
use reinforcement_learning::{
    action_selection::AdaptativeEpsilon,
    agent::{ContinuousObsDiscreteActionAgent, DoubleDeepAgent},
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
            &mem_policy.root() / "input",
            4,
            8,
            Default::default(),
        ))
        .add_fn(|xs| xs.tanh())
        .add(nn::linear(
            &mem_policy.root() / "output",
            8,
            2,
            Default::default(),
        ));
    (Box::new(policy_net), mem_policy)
}

fn main() {
    let seed: u64 = 42;
    tch::manual_seed(seed as i64);
    tch::maybe_init_cuda();
    const MEM_SIZE: usize = 10_000;
    const MIN_MEM_SIZE: usize = 1_000;
    const GAMMA: f32 = 0.95;
    const UPDATE_FREQ: usize = 10;
    const LEARNING_RATE: f64 = 0.005;

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

    let mut curr_obs = {
        let s = env.reset();
        Array1::from_vec(vec![
            s.cart_position,
            s.cart_velocity,
            s.pole_angle,
            s.pole_angular_velocity,
        ])
    };
    let mut curr_action;
    let mut next_obs;
    let mut reward;
    let mut terminated;

    let mut ep_return: f32 = 0.0;
    let mut n_episodes = 0;

    let trainer = ContinuousObsDiscreteTrainer::new(
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

    for step in 0..100_000 {
        curr_action = agent.get_action(&curr_obs);
        (next_obs, reward, terminated) = {
            let (state_, reward, done) = env.step(curr_action).unwrap();
            (
                Array1::from_vec(vec![
                    state_.cart_position,
                    state_.cart_velocity,
                    state_.pole_angle,
                    state_.pole_angular_velocity,
                ]),
                reward,
                done,
            )
        };
        let _td = agent.update(
            &curr_obs,
            curr_action,
            reward,
            terminated,
            &next_obs,
            curr_action,
        );
        // agent.memory.add(
        //     &curr_obs.try_into().unwrap(),
        //     curr_action as i64,
        //     reward,
        //     terminated,
        //     &next_obs.clone().try_into().unwrap(),
        //     curr_action as i64,
        // );

        // if agent.memory.ready() {
        //     let (b_state, b_action, b_reward, b_done, b_state_, _b_actions_) =
        //         agent.memory.sample_batch(128);
        //     let qvalues = agent.policy.forward(&b_state).gather(1, &b_action, false);

        //     let max_target_values: Tensor =
        //         tch::no_grad(|| agent.target_policy.forward(&b_state_).max_dim(1, true).0);
        //     let expected_values =
        //         b_reward + GAMMA * (&Tensor::from(1.0) - &b_done) * (&max_target_values);
        //     let expected_values = expected_values.to_kind(Kind::Float).to_device(device);
        //     let loss = qvalues.mse_loss(&expected_values, tch::Reduction::Mean);
        //     agent.optimizer.zero_grad();
        //     loss.backward();
        //     agent.optimizer.step();
        //     if n_episodes % UPDATE_FREQ == 0 {
        //         let a = agent.target_policy_vs.copy(&agent.policy_vs);
        //         if a.is_err() {
        //             println!("copy error")
        //         };
        //     }
        // }
        ep_return += reward;
        curr_obs = next_obs;

        if terminated {
            if n_episodes % 100 == 0 {
                println!(
                    "Episode: {}, step {} - last Return: {}",
                    n_episodes, step, ep_return
                );
                let (r1, _) = trainer.evaluate(&mut env, &mut agent, 100);
                let r1_m = r1.iter().sum::<f32>() / r1.len() as f32;
                if r1_m >= 475.0 {
                    println!(
                        "Solved at episode {} step {} with evaluation of {}",
                        n_episodes, step, r1_m
                    );
                    break;
                }
            }
            n_episodes += 1;
            ep_return = 0.0;
            curr_obs = {
                let s = env.reset();
                Array::from_vec(vec![
                    s.cart_position,
                    s.cart_velocity,
                    s.pole_angle,
                    s.pole_angular_velocity,
                ])
            };
        }
    }
    let (r1, _) = trainer.evaluate(&mut env, &mut agent, 100);
    println!(
        "steps limit reached. final evaluation of {}",
        r1.iter().sum::<f32>() / r1.len() as f32
    );
}
