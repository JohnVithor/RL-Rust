use std::{rc::Rc, time::Instant};

extern crate environments;
extern crate reinforcement_learning;
extern crate structopt;

use environments::toy_text::TaxiEnv;
use reinforcement_learning::{
    action_selection::{UniformEpsilonGreed, UpperConfidenceBound},
    agent::{expected_sarsa, one_step_agent::OneStepAgent, qlearning, sarsa, DiscreteAgent},
    trainer::DiscreteTrainer,
};
use serde_json::json;
use structopt::StructOpt;

use samples::{moving_average, save_json};

/// Train four RL agents using some parameters and generate some graphics of their results
#[derive(StructOpt, Debug)]
#[structopt(name = "RLRust - Taxi")]
struct Cli {
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

    let mut env = TaxiEnv::new(max_steps);

    let mut train_rewards: Vec<Vec<f64>> = vec![];
    let mut train_episodes_length: Vec<Vec<f64>> = vec![];
    let mut train_errors: Vec<Vec<f64>> = vec![];
    let mut test_rewards: Vec<Vec<f64>> = vec![];
    let mut test_episodes_length: Vec<Vec<f64>> = vec![];

    let mut epsilon_sarsa_agent = OneStepAgent::new(
        Box::new(UniformEpsilonGreed::new(
            initial_epsilon,
            Rc::new(move |a| a - epsilon_decay),
            final_epsilon,
        )),
        sarsa,
        learning_rate,
        0.0,
        discount_factor,
    );

    let mut epsilon_qlearning_agent = OneStepAgent::new(
        Box::new(UniformEpsilonGreed::new(
            initial_epsilon,
            Rc::new(move |a| a - epsilon_decay),
            final_epsilon,
        )),
        qlearning,
        learning_rate,
        0.0,
        discount_factor,
    );

    let mut epsilon_expected_sarsa_agent = OneStepAgent::new(
        Box::new(UniformEpsilonGreed::new(
            initial_epsilon,
            Rc::new(move |a| a - epsilon_decay),
            final_epsilon,
        )),
        expected_sarsa,
        learning_rate,
        0.0,
        discount_factor,
    );

    let mut ucb_sarsa_agent = OneStepAgent::new(
        Box::new(UpperConfidenceBound::new(confidence_level)),
        sarsa,
        learning_rate,
        0.0,
        discount_factor,
    );

    let mut ucb_qlearning_agent = OneStepAgent::new(
        Box::new(UpperConfidenceBound::new(confidence_level)),
        qlearning,
        learning_rate,
        0.0,
        discount_factor,
    );

    let mut ucb_expected_sarsa_agent = OneStepAgent::new(
        Box::new(UpperConfidenceBound::new(confidence_level)),
        expected_sarsa,
        learning_rate,
        0.0,
        discount_factor,
    );

    let agents: Vec<&mut OneStepAgent> = vec![
        &mut epsilon_sarsa_agent,
        &mut epsilon_qlearning_agent,
        &mut epsilon_expected_sarsa_agent,
        &mut ucb_sarsa_agent,
        &mut ucb_qlearning_agent,
        &mut ucb_expected_sarsa_agent,
    ];
    let identifiers = [
        "ε-Greedy One-Step Sarsa",
        "ε-Greedy One-Step Qlearning",
        "ε-Greedy One-Step Expected Sarsa",
        "UCB One-Step Sarsa",
        "UCB One-Step Qlearning",
        "UCB One-Step Expected Sarsa",
        // "ε-Greedy Trace Sarsa",
        // "ε-Greedy Trace Qlearning",
        // "ε-Greedy Trace Expected Sarsa",
        // "UCB Trace Sarsa",
        // "UCB Trace Qlearning",
        // "UCB Trace Expected Sarsa",
    ];

    for (i, agent) in agents.into_iter().enumerate() {
        println!("{} has:", identifiers[i]);
        let mut trainer = DiscreteTrainer::new(
            |repr: usize| -> usize { repr },
            |repr: &usize| -> usize { *repr },
        );
        let now: Instant = Instant::now();
        let (
            training_reward,
            training_length,
            training_error,
            _evaluation_reward,
            _evaluation_length,
        ) = trainer.train(&mut env, agent, n_episodes, n_episodes / 10, 100);
        let elapsed: std::time::Duration = now.elapsed();
        println!(" - training time of {:.2?}", elapsed);

        let ma_error = moving_average(
            training_error.len() / moving_average_window,
            &training_error,
        );
        train_errors.push(ma_error);
        let ma_reward = moving_average(
            n_episodes as usize / moving_average_window,
            &training_reward,
        );
        let v: Vec<f64> = training_length.iter().map(|x| *x as f64).collect();
        train_rewards.push(ma_reward);
        let ma_episode = moving_average(n_episodes as usize / moving_average_window, &v);
        train_episodes_length.push(ma_episode);

        let (testing_rewards, testing_length) = trainer.evaluate(&mut env, agent, n_episodes);

        let ma_reward = moving_average(
            n_episodes as usize / moving_average_window,
            &testing_rewards,
        );
        test_rewards.push(ma_reward);
        let v: Vec<f64> = testing_length.iter().map(|x| *x as f64).collect();
        let ma_episode = moving_average(n_episodes as usize / moving_average_window, &v);
        test_episodes_length.push(ma_episode);

        let mut mean_reward = 0.0;
        for v in testing_rewards {
            mean_reward += v;
        }
        mean_reward /= n_episodes as f64;
        println!(" - mean reward of {:.2?}", mean_reward);
        agent.reset();
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
