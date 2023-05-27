use std::cell::RefCell;
use std::time::Instant;

use plotters::prelude::{BitMapBackend, ChartBuilder, IntoDrawingArea, LabelAreaPosition};
use plotters::series::LineSeries;
use plotters::style::{BLUE, WHITE};

use reinforcement_learning::algorithms::action_selection::{UniformEpsilonGreed, UpperConfidenceBound};
use reinforcement_learning::algorithms::policy_update::{QStep, SarsaStep, SarsaLambda, QLambda};
use reinforcement_learning::env::{Env, EnvNotReady, BlackJackEnv, FrozenLakeEnv, CliffWalkingEnv, TaxiEnv};
use reinforcement_learning::utils::moving_average;
use reinforcement_learning::{Agent, Policy};

fn main() {
    let learning_rate: f64 = 0.05;
    let n_episodes: u128 = 100_000;
    let start_epsilon: f64 = 1.0;
    let epsilon_decay: f64 = start_epsilon / (n_episodes as f64 / 2.0);
    let final_epsilon: f64 = 0.0;
    let confidence_level: f64 = 0.8;
    let discount_factor: f64 = 0.95;
    let lambda_factor: f64 = 0.5;
    let max_steps: u128 = 100;
    let mut env = BlackJackEnv::new();
    // let mut env = FrozenLakeEnv::new(&FrozenLakeEnv::MAP_4X4, true, max_steps);
    // let mut env = CliffWalkingEnv::new(max_steps);
    // let mut env = TaxiEnv::new(max_steps);

    println!("Play!");
    env.play();

    let policy_update_strategy = QLambda::new(learning_rate, discount_factor, lambda_factor, 0.0, env.action_space());
    // let policy_update_strategy = QStep::new(learning_rate, discount_factor);
    let action_selection_strategy = UniformEpsilonGreed::new(start_epsilon, epsilon_decay, final_epsilon);
    // let action_selection_strategy = UpperConfidenceBound::new(confidence_level);
    let policy = Policy::new(0.0, env.action_space());

    let agent = &mut Agent::new(
        Box::new(RefCell::new(action_selection_strategy)),
        Box::new(RefCell::new(policy_update_strategy)),
        policy,
        env.action_space(),
    );

    println!("Training agent!");
    let now: Instant = Instant::now();
    let reward_history: Vec<f64> = agent.train(&mut env, n_episodes);
    let elapsed: std::time::Duration = now.elapsed();
    println!("Training done!");
    println!("Time elapsed: {:.2?}", elapsed);

    let values_moving_average: Vec<f64> = moving_average(n_episodes as usize / 100, &reward_history);
    let min_reward = values_moving_average.iter().copied().reduce(f64::min).unwrap();
    let max_reward = values_moving_average.iter().copied().reduce(f64::max).unwrap();

    println!("mininum average reward: {:?}, maximun average reward: {:?}", min_reward, max_reward);

    let root_area = BitMapBackend::new("test.png", (600, 400)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let mut ctx = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption("Episode reward", ("sans-serif", 40))
        .build_cartesian_2d(0..values_moving_average.len(), min_reward..max_reward)
        .unwrap();

    ctx.configure_mesh().draw().unwrap();

    ctx.draw_series(LineSeries::new(
        (0..)
            .zip(values_moving_average.iter())
            .map(|(idx, y)| (idx, *y)),
        &BLUE,
    ))
    .unwrap();


    // let mut epi_reward = 0.0;
    // let mut curr_obs = env.reset();
    // let mut curr_action: usize = agent.get_action(&curr_obs);
    // let mut steps: i32 = 0;
    // for _i in 0..100 {
    //     steps+=1;
    //     match env.step(curr_action) {
    //         Ok((next_obs, reward, terminated)) => {
    //             let next_action: usize = agent.get_action(&next_obs);
    //             println!("curr_obs {:?}", curr_obs);
    //             println!("curr_action {:?}", curr_action);
    //             println!("reward {:?}", reward);
    //             curr_obs = next_obs;
    //             curr_action = next_action;
    //             epi_reward+=reward;
    //             if terminated {
    //                 println!("curr_obs {:?}", curr_obs);
    //                 println!("episode reward {:?}", epi_reward);
    //                 println!("terminated with {:?} steps", steps);
    //                 break;
    //             }
    //         }
    //         Err(EnvNotReady) => {
    //             println!("Environment is not ready to receive actions!");
    //             break;
    //         }
    //     }
    // }

}
