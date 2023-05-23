use std::cell::RefCell;
use std::time::Instant;

use plotters::prelude::{BitMapBackend, ChartBuilder, IntoDrawingArea, LabelAreaPosition};
use plotters::series::LineSeries;
use plotters::style::{BLUE, WHITE};

use reinforcement_learning::algorithms::{action_selection::EpsilonGreed, policy_update::QStep};
use reinforcement_learning::env::{BlackJackEnv, Env, EnvNotReady, Observation};
use reinforcement_learning::utils::moving_average;
use reinforcement_learning::{Agent, Policy};

fn main() {
    let learning_rate: f64 = 0.05;
    let n_episodes: i32 = 100_000;
    let start_epsilon: f64 = 1.0;
    let epsilon_decay: f64 = start_epsilon / (n_episodes as f64 / 2.0);
    let final_epsilon: f64 = 0.0;
    let discount_factor: f64 = 0.95;
    let mut env: BlackJackEnv = BlackJackEnv::new();

    let policy_update_strategy = QStep::new(learning_rate, discount_factor);
    let action_selection_strategy = EpsilonGreed::new(start_epsilon, epsilon_decay, final_epsilon);
    let policy: Policy = Policy::new(0.0, env.action_space());

    let agent: &mut Agent = &mut Agent::new(
        Box::new(RefCell::new(action_selection_strategy)),
        Box::new(RefCell::new(policy_update_strategy)),
        policy,
        env.action_space(),
    );

    let mut reward_history: Vec<f64> = vec![];

    let now: Instant = Instant::now();
    for _episode in 0..n_episodes {
        let mut curr_obs: Observation = env.reset();
        let mut curr_action: usize = agent.get_action(&curr_obs);
        loop {
            match env.step(curr_action) {
                Ok((next_obs, reward, terminated)) => {
                    let next_action: usize = agent.get_action(&next_obs);
                    agent.update(
                        &curr_obs,
                        curr_action,
                        reward,
                        terminated,
                        &next_obs,
                        next_action,
                    );
                    curr_obs = next_obs;
                    curr_action = next_action;
                    if terminated {
                        reward_history.push(reward);
                        break;
                    }
                }
                Err(EnvNotReady) => {
                    println!("Ambiente não está pronto para receber ações!");
                    break;
                }
            }
        }
    }
    let elapsed: std::time::Duration = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);

    let values_moving_average: Vec<f64> =
        moving_average(n_episodes as usize / 100, &reward_history);

    let root_area = BitMapBackend::new("test.png", (600, 400)).into_drawing_area();

    root_area.fill(&WHITE).unwrap();

    let mut ctx = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption("Episode reward", ("sans-serif", 40))
        .build_cartesian_2d(0..values_moving_average.len(), -1.0..1.0)
        .unwrap();

    ctx.configure_mesh().draw().unwrap();

    ctx.draw_series(LineSeries::new(
        (0..)
            .zip(values_moving_average.iter())
            .map(|(idx, y)| (idx, *y)),
        &BLUE,
    ))
    .unwrap();
}
