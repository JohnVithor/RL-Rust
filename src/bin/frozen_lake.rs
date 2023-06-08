use std::rc::Rc;
use std::time::Instant;

use plotters::style::{RGBColor, BLUE, CYAN, GREEN, MAGENTA, RED, YELLOW};

use reinforcement_learning::action_selection::{
    EnumActionSelection, UniformEpsilonGreed, UpperConfidenceBound,
};
use reinforcement_learning::agent::{expected_sarsa, qlearning, sarsa};
use reinforcement_learning::agent::{Agent, ElegibilityTracesAgent, OneStepAgent};
use reinforcement_learning::env::FrozenLakeEnv;
use reinforcement_learning::policy::{EnumPolicy, TabularPolicy};
use reinforcement_learning::utils::{moving_average, plot_moving_average};

extern crate structopt;

use structopt::StructOpt;

/// Train four RL agents using some parameters and generate some graphics of their results
#[derive(StructOpt, Debug)]
#[structopt(name = "RLRust - FrozenLake")]
struct Cli {
    /// Should the env be stochastic
    #[structopt(long = "stochastic_env")]
    stochastic_env: bool,

    /// Change the env's map, if possible
    #[structopt(long = "map", default_value = "4x4")]
    map: String,

    /// Show example of episode
    #[structopt(long = "show_example")]
    show_example: bool,

    /// Number of episodes for the training
    #[structopt(long = "n_episodes", short = "n", default_value = "100000")]
    n_episodes: u128,

    /// Maximum number of steps per episode
    #[structopt(long = "max_steps", default_value = "100")]
    max_steps: u128,

    /// Learning rate of the RL agent
    #[structopt(long = "learning_rate", default_value = "0.05")]
    learning_rate: f64,

    /// Initial value for the exploration ratio
    #[structopt(long = "initial_epsilon", default_value = "1.0")]
    initial_epsilon: f64,

    /// Value to determine percentage of episodes where exploration is possible;
    #[structopt(long = "exploration_time", default_value = "0.5")]
    exploration_time: f64,

    /// Final value for the exploration ratio
    #[structopt(long = "final_epsilon", default_value = "0.0")]
    final_epsilon: f64,

    /// Confidence level for the UCB action selection strategy
    #[structopt(long = "confidence_level", default_value = "0.5")]
    confidence_level: f64,

    /// Discont factor to be used on the temporal difference calculation
    #[structopt(long = "discount_factor", default_value = "0.95")]
    discount_factor: f64,

    /// Lambda factor to be used on the eligibility traces algorithms
    #[structopt(long = "lambda_factor", default_value = "0.5")]
    lambda_factor: f64,

    /// Moving average window to be used on the visualization of results
    #[structopt(long = "moving_average_window", default_value = "100")]
    moving_average_window: usize,
}

fn main() {
    let cli: Cli = Cli::from_args();

    let n_episodes: u128 = cli.n_episodes;
    let max_steps: u128 = cli.max_steps;

    let learning_rate: f64 = cli.learning_rate;
    let initial_epsilon: f64 = cli.initial_epsilon;
    let epsilon_decay: f64 = initial_epsilon / (cli.exploration_time * n_episodes as f64);
    let final_epsilon: f64 = cli.final_epsilon;
    let confidence_level: f64 = cli.confidence_level;
    let discount_factor: f64 = cli.discount_factor;
    let lambda_factor: f64 = cli.lambda_factor;

    let moving_average_window: usize = cli.moving_average_window;

    let mut env = FrozenLakeEnv::new(
        if cli.map == "4x4" {
            &FrozenLakeEnv::MAP_4X4
        } else {
            &FrozenLakeEnv::MAP_8X8
        },
        cli.stochastic_env,
        max_steps,
    );

    let mut train_rewards: Vec<Vec<f64>> = vec![];
    let mut train_episodes_length: Vec<Vec<f64>> = vec![];
    let mut train_errors: Vec<Vec<f64>> = vec![];
    let mut test_rewards: Vec<Vec<f64>> = vec![];
    let mut test_episodes_length: Vec<Vec<f64>> = vec![];

    const SIZE: usize = 4;

    let legends: Vec<&str> = [
        "ε-Greedy One-Step Sarsa",
        "ε-Greedy One-Step Qlearning",
        "ε-Greedy One-Step Expected Sarsa",
        "UCB One-Step Sarsa",
        "UCB One-Step Qlearning",
        "UCB One-Step Expected Sarsa",
        "ε-Greedy Trace Sarsa",
        "ε-Greedy Trace Qlearning",
        "ε-Greedy Trace Expected Sarsa",
        "UCB Trace Sarsa",
        "UCB Trace Qlearning",
        "UCB Trace Expected Sarsa",
    ]
    .to_vec();

    const DRED: RGBColor = RGBColor(150, 0, 0);
    const DBLUE: RGBColor = RGBColor(0, 0, 150);
    const DGREEN: RGBColor = RGBColor(0, 150, 0);

    const DDRED: RGBColor = RGBColor(50, 0, 0);
    const DDBLUE: RGBColor = RGBColor(0, 0, 50);
    const DDGREEN: RGBColor = RGBColor(0, 50, 0);

    let colors: Vec<&plotters::style::RGBColor> = [
        &BLUE, &GREEN, &CYAN, &RED, &YELLOW, &MAGENTA, &DRED, &DBLUE, &DGREEN, &DDRED, &DDBLUE,
        &DDGREEN,
    ]
    .to_vec();

    let policy = TabularPolicy::new(0.0);
    // let policy = DoubleTabularPolicy::new(0.0);

    let action_selection = vec![
        EnumActionSelection::from(UniformEpsilonGreed::new(
            initial_epsilon,
            Rc::new(move |a| a - epsilon_decay),
            final_epsilon,
        )),
        EnumActionSelection::from(UpperConfidenceBound::new(confidence_level)),
    ];

    let mut one_step_agent: OneStepAgent<usize, SIZE> = OneStepAgent::new(
        EnumPolicy::from(policy.clone()),
        learning_rate,
        discount_factor,
        action_selection[0].clone(),
        sarsa,
    );

    let mut trace_agent: ElegibilityTracesAgent<usize, SIZE> =
        ElegibilityTracesAgent::new(
            EnumPolicy::from(policy),
            learning_rate,
            discount_factor,
            action_selection[0].clone(),
            lambda_factor,
            sarsa,
        );

    let mut agents: Vec<&mut dyn Agent<usize, SIZE>> = vec![];
    agents.push(&mut one_step_agent);
    agents.push(&mut trace_agent);

    let mut i = 0;
    for agent in agents {
        for acs in &action_selection {
            agent.set_action_selector(acs.clone());
            for func in [sarsa, qlearning, expected_sarsa] {
                agent.set_future_q_value_func(func);
                let now: Instant = Instant::now();
                let (reward_history, episode_length, training_error) =
                    agent.train(&mut env, n_episodes, n_episodes / 10);
                let elapsed: std::time::Duration = now.elapsed();
                println!("{} {:.2?}", legends[i], elapsed);

                let ma_error = moving_average(
                    training_error.len() / moving_average_window,
                    &training_error,
                );
                train_errors.push(ma_error);
                let ma_reward =
                    moving_average(n_episodes as usize / moving_average_window, &reward_history);
                train_rewards.push(ma_reward);
                let ma_episode = moving_average(
                    n_episodes as usize / moving_average_window,
                    &episode_length.iter().map(|x| *x as f64).collect(),
                );
                train_episodes_length.push(ma_episode);

                if cli.show_example {
                    agent.example(&mut env);
                }

                let (reward_history, episode_length) = agent.evaluate(&mut env, n_episodes);

                let ma_reward =
                    moving_average(n_episodes as usize / moving_average_window, &reward_history);
                test_rewards.push(ma_reward);
                let ma_episode = moving_average(
                    n_episodes as usize / moving_average_window,
                    &episode_length.iter().map(|x| *x as f64).collect(),
                );
                test_episodes_length.push(ma_episode);

                i += 1;
                agent.reset();
            }
        }
    }
    plot_moving_average(&train_rewards, &colors, &legends, "Train Rewards");

    plot_moving_average(
        &train_episodes_length,
        &colors,
        &legends,
        "Train Episodes Length",
    );

    plot_moving_average(&train_errors, &colors, &legends, "Training Error");

    plot_moving_average(&test_rewards, &colors, &legends, "Test Rewards");

    plot_moving_average(
        &test_episodes_length,
        &colors,
        &legends,
        "Test Episodes Length",
    );
}
