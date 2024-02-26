use environments::{
    classic_control::{cart_pole::CartPoleObservation, CartPoleEnv},
    Env,
};
use ndarray::{Array, Array1};
use reinforcement_learning::{
    action_selection::AdaptativeEpsilon, agent::DoubleDeepAgent,
    trainer::ContinuousObsDiscreteTrainer,
};
use tch::{
    nn::{self, Adam, Module, VarStore},
    Device, Kind, Tensor,
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
            128,
            Default::default(),
        ))
        .add_fn(|xs| xs.tanh())
        .add(nn::linear(
            &mem_policy.root() / "al3",
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

    let mut state = {
        let s = env.reset();
        Array1::from_vec(vec![
            s.cart_position,
            s.cart_velocity,
            s.pole_angle,
            s.pole_angular_velocity,
        ])
    };
    let mut action;
    let mut state_;
    let mut reward;
    let mut done;

    let mut ep_return: f32 = 0.0;
    let mut nepisodes = 0;
    let one: Tensor = Tensor::from(1.0);

    loop {
        let values = tch::no_grad(|| {
            agent
                .policy
                .forward(&Tensor::try_from(state.clone()).unwrap().to_device(device))
        });
        let values: ndarray::ArrayD<f32> = (&values).try_into().unwrap();
        let values_len: usize = values.len();
        action = agent
            .action_selection
            .get_action(&values.into_shape(values_len).unwrap());
        (state_, reward, done) = {
            let (state_, reward, done) = env.step(action).unwrap();
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
        agent.memory.add(
            &state.try_into().unwrap(),
            action as i64,
            reward,
            done,
            &state_.clone().try_into().unwrap(),
            action as i64,
        );
        ep_return += reward;
        state = state_;

        if done {
            state = {
                let s = env.reset();
                Array::from_vec(vec![
                    s.cart_position,
                    s.cart_velocity,
                    s.pole_angle,
                    s.pole_angular_velocity,
                ])
            };

            if nepisodes % 100 == 0 {
                println!("Episode: {}, last Return: {}", nepisodes, ep_return);
            }
            if ep_return >= 475.0 {
                println!(
                    "Solved at episode {} with last return of {}",
                    nepisodes, ep_return
                );

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
                let (r1, r2) = trainer.evaluate(&mut env, &mut agent, 100);
                println!(
                    "{} - {}",
                    r1.iter().sum::<f32>() / r1.len() as f32,
                    r2.iter().sum::<u128>() / r2.len() as u128
                );
                break;
            }
            agent.action_selection.update(ep_return);
            nepisodes += 1;
            ep_return = 0.0;
        }

        if agent.memory.ready() {
            let (b_state, b_action, b_reward, b_done, b_state_, _b_actions_) =
                agent.memory.sample_batch(128);
            let qvalues = agent.policy.forward(&b_state).gather(1, &b_action, false);

            let max_target_values: Tensor =
                tch::no_grad(|| agent.target_policy.forward(&b_state_).max_dim(1, true).0);
            let expected_values = b_reward + GAMMA * (&one - &b_done) * (&max_target_values);
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
}
