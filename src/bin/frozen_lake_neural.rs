use std::rc::Rc;
use std::time::Instant;
use std::vec;

use ndarray::arr2;
use plotters::style::{RGBColor, BLUE, CYAN, GREEN, MAGENTA, RED, YELLOW};

use reinforcement_learning::action_selection::{EnumActionSelection, UniformEpsilonGreed};
use reinforcement_learning::agent::Agent;
use reinforcement_learning::agent::{qlearning, OneStepAgent};
use reinforcement_learning::env::FrozenLakeEditedEnv;
use reinforcement_learning::env::frozen_lake_edited::FrozenLakeObs;
use reinforcement_learning::network::activation::{tanh, tanh_prime};
use reinforcement_learning::network::layers::{ActivationLayer, DenseLayer};
use reinforcement_learning::network::loss::{mse, mse_prime};
use reinforcement_learning::network::Network;
use reinforcement_learning::policy::{EnumPolicy, NeuralPolicy};
use reinforcement_learning::utils::{moving_average, plot_moving_average};

extern crate structopt;

use structopt::StructOpt;

/// Train four RL agents using some parameters and generate some graphics of their results
#[derive(StructOpt, Debug)]
#[structopt(name = "RLRust - Frozen lake - neural")]
struct Cli {
    /// Show example of episode
    #[structopt(long = "show_example", short = "s")]
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
    let epsilon_decay: f64 = cli.exploration_time;
    let final_epsilon: f64 = cli.final_epsilon;
    let _confidence_level: f64 = cli.confidence_level;
    let discount_factor: f64 = cli.discount_factor;
    let _lambda_factor: f64 = cli.lambda_factor;

    let moving_average_window: usize = cli.moving_average_window;

    let mut env = FrozenLakeEditedEnv::new(&FrozenLakeEditedEnv::MAP_4X4, false, max_steps);

    let mut train_rewards: Vec<Vec<f64>> = vec![];
    let mut train_episodes_length: Vec<Vec<f64>> = vec![];
    let mut train_errors: Vec<Vec<f64>> = vec![];
    let mut test_rewards: Vec<Vec<f64>> = vec![];
    let mut test_episodes_length: Vec<Vec<f64>> = vec![];

    const SIZE: usize = 4;

    let legends: Vec<&str> = [
        // "ε-Greedy One-Step Sarsa",
        "ε-Greedy One-Step Qlearning",
        "ε-Greedy One-Step Dyna-Qlearning",
        // "ε-Greedy One-Step Expected Sarsa",
        // "UCB One-Step Sarsa",
        // "UCB One-Step Qlearning",
        // "UCB One-Step Expected Sarsa",
        // "ε-Greedy Trace Sarsa",
        // "ε-Greedy Trace Qlearning",
        // "ε-Greedy Trace Expected Sarsa",
        // "UCB Trace Sarsa",
        // "UCB Trace Qlearning",
        // "UCB Trace Expected Sarsa",
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

    let mut network = Network::new(learning_rate, mse, mse_prime);
    network.add(Box::new(DenseLayer::new(6, 20)));
    network.add(Box::new(ActivationLayer::new(tanh, tanh_prime)));
    network.add(Box::new(DenseLayer::new(20, 20)));
    network.add(Box::new(ActivationLayer::new(tanh, tanh_prime)));
    network.add(Box::new(DenseLayer::new(20, 4)));
    network.add(Box::new(ActivationLayer::new(tanh, tanh_prime)));

    fn input_adapter(obs: FrozenLakeObs) -> ndarray::Array2<f64> {
        arr2(&[[
            obs.left.value(),
            obs.down.value(),
            obs.right.value(),
            obs.up.value(),
            obs.x as f64,
            obs.y as f64,
        ]])
    }

    fn output_adapter(values: ndarray::Array2<f64>) -> [f64; 4] {
        [
            *values.get((0, 0)).unwrap(),
            *values.get((0, 1)).unwrap(),
            *values.get((0, 2)).unwrap(),
            *values.get((0, 3)).unwrap(),
        ]
    }

    fn inv_output_adapter(values: [f64; 4]) -> ndarray::Array2<f64> {
        arr2(&[[
            *values.get(0).unwrap(),
            *values.get(1).unwrap(),
            *values.get(2).unwrap(),
            *values.get(3).unwrap(),
        ]])
    }

    let policy = NeuralPolicy::new(input_adapter, network, output_adapter, inv_output_adapter);
    // let policy = DoubleTabularPolicy::new(0.0);

    let action_selection: Vec<EnumActionSelection<_, 4>> = vec![
        EnumActionSelection::from(UniformEpsilonGreed::new(
            initial_epsilon,
            Rc::new(move |a| {a * epsilon_decay}),
            final_epsilon,
        )),
        // EnumActionSelection::from(UpperConfidenceBound::new(confidence_level)),
    ];

    let mut one_step_agent: OneStepAgent<_, SIZE> = OneStepAgent::new(
        EnumPolicy::from(policy),
        learning_rate,
        discount_factor,
        action_selection[0].clone(),
        qlearning,
    );

    let mut agents: Vec<&mut dyn Agent<_, SIZE>> = vec![];
    agents.push(&mut one_step_agent);

    let mut i = 0;
    for agent in agents {
        for acs in &action_selection {
            agent.set_action_selector(acs.clone());
            let now: Instant = Instant::now();
            let (reward_history, episode_length, training_error) =
                agent.train(&mut env, n_episodes, n_episodes/10);
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
