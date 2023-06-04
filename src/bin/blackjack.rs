use std::time::Instant;

use plotters::style::{BLUE, GREEN, RED, YELLOW, CYAN, MAGENTA};

use reinforcement_learning::agent::train;
use reinforcement_learning::env::{Env, BlackJackEnv, FrozenLakeEnv, CliffWalkingEnv, TaxiEnv};
use reinforcement_learning::utils::{moving_average, plot_moving_average};
use reinforcement_learning::{agent::OneStepTabularEGreedySarsa};

extern crate structopt;

use structopt::StructOpt;

/// Train four RL agents using some parameters and generate some graphics of their results
#[derive(StructOpt, Debug)]
#[structopt(name = "RLRust - BlackJack")]
struct Cli {

    /// Show example of episode
    #[structopt(long = "show_example")]
    show_example: bool,

    /// Number of episodes for the training
    #[structopt(long = "n_episodes", default_value = "100000")]
    n_episodes: u128,

    /// Maximum number of steps per episode
    #[structopt(long = "max_steps", default_value = "100")]
    max_steps: u128,

    /// Learning rate of the RL agent
    #[structopt(long = "learning_rate", default_value = "0.05")]
    learning_rate: f64,

    /// Action selection strategy: 0 - uniform epsilon greed, 1 - upper confidence bound
    #[structopt(long = "action_selection", default_value = "0")]
    action_selection: u8,

    /// Policy type: 0 - basic policy, 1 - double policy
    #[structopt(long = "policy_type", default_value = "0")]
    policy_type: u8,

    /// Initial value for the exploration ratio
    #[structopt(long = "initial_epsilon", default_value = "1.0")]
    initial_epsilon: f64,

    /// Value to decrease of the exploration ratio at each step default to: initial_epsilon / (n_episodes / 2);
    #[structopt(long = "epsilon_decay", default_value = "NAN")]
    epsilon_decay: f64,

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
    let epsilon_decay: f64 = if cli.epsilon_decay.is_nan() {initial_epsilon / (9.0 * n_episodes as f64 / 10.0)} else {cli.epsilon_decay};
    let final_epsilon: f64 = cli.final_epsilon;
    let confidence_level: f64 = cli.confidence_level;
    let discount_factor: f64 = cli.discount_factor;
    let lambda_factor: f64 = cli.lambda_factor;
    
    let moving_average_window: usize = cli.moving_average_window;

    let mut env = BlackJackEnv::new();
    
    let mut rewards: Vec<Vec<f64>> = vec![];
    let mut episodes_length: Vec<Vec<f64>> = vec![];
    let mut errors : Vec<Vec<f64>> = vec![];

    let legends: Vec<&str> = [
        "One-Step Sarsa",
        // "One-Step Qlearning",
        // "One-Step Expected Sarsa",
        // "Sarsa Lambda",
        // "Qlearning Lambda",
    ].to_vec();

    let colors: Vec<&plotters::style::RGBColor> = [
        &BLUE,
        // &GREEN,
        // &CYAN,
        // &RED,
        // &YELLOW
    ].to_vec();

    let mut agent: OneStepTabularEGreedySarsa<usize, 2> = OneStepTabularEGreedySarsa::new(
        0.0,
        learning_rate,
        discount_factor,
        initial_epsilon,
        epsilon_decay,
        final_epsilon
    );


    let now: Instant = Instant::now();
    let (reward_history, episode_length)  = train(&mut agent, &mut env, n_episodes);
    let elapsed: std::time::Duration = now.elapsed();
    println!("{} {:.2?}", "sarsa", elapsed);

    let ma_error = moving_average(agent.get_training_error().len() as usize / moving_average_window, &agent.get_training_error());
    errors.push(ma_error);
    let ma_reward = moving_average(n_episodes as usize / moving_average_window, &reward_history);
    rewards.push(ma_reward);
    let ma_episode = moving_average(n_episodes as usize / moving_average_window, &episode_length.iter().map(|x| *x as f64).collect());
    episodes_length.push(ma_episode);

    // if cli.show_example {
    //     agent.example(env);
    // }
    
    plot_moving_average(
        &rewards,
        &colors,
        &legends,
        "Rewards"
    );

    plot_moving_average(
        &episodes_length,
        &colors,
        &legends,
        "Episodes Length"
    );

    plot_moving_average(
        &errors,
        &colors,
        &legends,
        "Training Error"
    );
}
