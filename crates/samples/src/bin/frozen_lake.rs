use std::rc::Rc;
use std::time::Instant;

extern crate environments;
extern crate reinforcement_learning;
extern crate structopt;

use environments::toy_text::frozen_lake::{FrozenLakeAction, FrozenLakeEnv};
use reinforcement_learning::action_selection::{UniformEpsilonGreed, UpperConfidenceBound};
use reinforcement_learning::agent::{expected_sarsa, qlearning, sarsa};
use reinforcement_learning::agent::{DiscreteAgent, ElegibilityTracesAgent, OneStepAgent};
use reinforcement_learning::policy::TabularPolicy;
use samples::{moving_average, save_json};
use serde_json::json;
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
    #[structopt(long = "exploration_time", default_value = "0.9")]
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

    let mut one_step_policy_epg = TabularPolicy::new(learning_rate, 0.0);
    let mut one_step_policy_ucb = TabularPolicy::new(learning_rate, 0.0);
    let mut trace_policy_epg = TabularPolicy::new(learning_rate, 0.0);
    let mut trace_policy_ucb = TabularPolicy::new(learning_rate, 0.0);

    let mut one_step_epg = UniformEpsilonGreed::new(
        2,
        initial_epsilon,
        Rc::new(move |a| a - epsilon_decay),
        final_epsilon,
    );
    let mut one_step_ucb = UpperConfidenceBound::new(confidence_level);

    let mut trace_epg = UniformEpsilonGreed::new(
        2,
        initial_epsilon,
        Rc::new(move |a| a - epsilon_decay),
        final_epsilon,
    );
    let mut trace_ucb = UpperConfidenceBound::new(confidence_level);

    let mut one_step_agent_epg: OneStepAgent<usize, FrozenLakeAction> = OneStepAgent::new(
        discount_factor,
        sarsa::<FrozenLakeAction>,
        &mut one_step_policy_epg,
        &mut one_step_epg,
    );
    let mut one_step_agent_ucb: OneStepAgent<usize, FrozenLakeAction> = OneStepAgent::new(
        discount_factor,
        sarsa::<FrozenLakeAction>,
        &mut one_step_policy_ucb,
        &mut one_step_ucb,
    );

    let mut trace_agent_epg: ElegibilityTracesAgent<usize, FrozenLakeAction> =
        ElegibilityTracesAgent::new(
            discount_factor,
            lambda_factor,
            sarsa::<FrozenLakeAction>,
            &mut trace_policy_epg,
            &mut trace_epg,
        );

    let mut trace_agent_ucb: ElegibilityTracesAgent<usize, FrozenLakeAction> =
        ElegibilityTracesAgent::new(
            discount_factor,
            lambda_factor,
            sarsa::<FrozenLakeAction>,
            &mut trace_policy_ucb,
            &mut trace_ucb,
        );

    let mut agents: Vec<&mut dyn DiscreteAgent<usize, FrozenLakeAction>> = vec![
        &mut one_step_agent_epg,
        &mut one_step_agent_ucb,
        &mut trace_agent_epg,
        &mut trace_agent_ucb,
    ];

    let identifiers = [
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
    ];

    let mut i = 0;
    for agent in agents.iter_mut() {
        for func in [
            sarsa::<FrozenLakeAction>,
            qlearning::<FrozenLakeAction>,
            expected_sarsa::<FrozenLakeAction>,
        ] {
            agent.set_future_q_value_func(func);
            println!("{}", identifiers[i]);
            let now: Instant = Instant::now();
            let (
                training_reward,
                training_length,
                training_error,
                _evaluation_reward,
                _evaluation_length,
            ) = agent.train(&mut env, n_episodes, n_episodes / 10, 100);
            let elapsed: std::time::Duration = now.elapsed();
            println!("{:.2?}", elapsed);

            let ma_error = moving_average(
                training_error.len() / moving_average_window,
                &training_error,
            );
            train_errors.push(ma_error);
            let ma_reward = moving_average(
                n_episodes as usize / moving_average_window,
                &training_reward,
            );
            train_rewards.push(ma_reward);
            let ma_episode = moving_average(
                n_episodes as usize / moving_average_window,
                &training_length.iter().map(|x| *x as f64).collect(),
            );
            train_episodes_length.push(ma_episode);

            let (testing_rewards, testing_length) = agent.evaluate(&mut env, n_episodes);

            let ma_reward = moving_average(
                n_episodes as usize / moving_average_window,
                &testing_rewards,
            );
            test_rewards.push(ma_reward);
            let ma_episode = moving_average(
                n_episodes as usize / moving_average_window,
                &testing_length.iter().map(|x| *x as f64).collect(),
            );
            test_episodes_length.push(ma_episode);

            i += 1;
            agent.reset();
        }
    }
    match save_json(
        "results.json",
        json!({
            "train_rewards": &train_rewards,
            "train_episodes_length": &train_episodes_length,
            "train_errors": &train_errors,
            "test_rewards": &test_rewards,
            "test_episodes_length": &test_episodes_length,
            "identifiers": &identifiers
        }),
    ) {
        Ok(_) => {
            println!("OK")
        }
        Err(_) => {
            println!("ERROR")
        }
    };
}
