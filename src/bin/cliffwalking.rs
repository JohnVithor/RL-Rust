use std::time::Instant;

use plotters::style::{BLUE, CYAN, GREEN, MAGENTA, RED, YELLOW, RGBColor};

use reinforcement_learning::action_selection::{UniformEpsilonGreed, UpperConfidenceBound};
use reinforcement_learning::agent::{expected_sarsa, qlearning, sarsa,OneStepTabularAgent};
use reinforcement_learning::agent::{
    Agent
};
use reinforcement_learning::env::CliffWalkingEnv;
use reinforcement_learning::utils::{moving_average, plot_moving_average};

extern crate structopt;

use structopt::StructOpt;

/// Train four RL agents using some parameters and generate some graphics of their results
#[derive(StructOpt, Debug)]
#[structopt(name = "RLRust - CliffWalking")]
struct Cli {

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

    let mut env = CliffWalkingEnv::new(max_steps);

    let mut rewards: Vec<Vec<f64>> = vec![];
    let mut episodes_length: Vec<Vec<f64>> = vec![];
    let mut errors: Vec<Vec<f64>> = vec![];

    const SIZE: usize = 4;

    let legends: Vec<&str> = [
        "ε-Greedy One-Step Sarsa",
        "ε-Greedy One-Step Qlearning",
        "ε-Greedy One-Step Expected Sarsa",
        // "ε-Greedy Trace Sarsa",
        // "ε-Greedy Trace Qlearning",
        // "ε-Greedy Trace Expected Sarsa",
        "UCB One-Step Sarsa",
        "UCB One-Step Qlearning",
        "UCB One-Step Expected Sarsa",
    ]
    .to_vec();

    const DRED: RGBColor = RGBColor(150, 0, 0);
    const DBLUE: RGBColor = RGBColor(0, 0, 150);
    const DGREEN: RGBColor = RGBColor(0, 150, 0);

    let colors: Vec<&plotters::style::RGBColor> = [
        // &BLUE, &GREEN, &CYAN, &RED, &YELLOW, &MAGENTA, &DRED, &DBLUE, &DGREEN,
        &BLUE, &GREEN, &CYAN, &RED, &YELLOW, &MAGENTA
    ]
    .to_vec();

    let mut aegreedy = UniformEpsilonGreed::new(initial_epsilon, epsilon_decay, final_epsilon);
    let mut begreedy = UniformEpsilonGreed::new(initial_epsilon, epsilon_decay, final_epsilon);
    let mut cegreedy = UniformEpsilonGreed::new(initial_epsilon, epsilon_decay, final_epsilon);

    let mut aucb = UpperConfidenceBound::new(confidence_level);
    let mut bucb = UpperConfidenceBound::new(confidence_level);
    let mut cucb = UpperConfidenceBound::new(confidence_level);

    let mut step_sarsa: OneStepTabularAgent<usize, SIZE> = OneStepTabularAgent::new(
        0.0,
        learning_rate,
        discount_factor,
        &mut aegreedy,
        sarsa,
    );

    let mut step_qlearning: OneStepTabularAgent<usize, SIZE> =
    OneStepTabularAgent::new(
            0.0,
            learning_rate,
            discount_factor,
            &mut begreedy,
            qlearning,
        );

    let mut step_expected_sarsa: OneStepTabularAgent<usize, SIZE> =
    OneStepTabularAgent::new(
            0.0,
            learning_rate,
            discount_factor,
            &mut cegreedy,
            expected_sarsa,
        );

    // let mut trace_sarsa: ElegibilityTracesTabularEGreedyAgent<usize, SIZE> =
    //     ElegibilityTracesTabularEGreedyAgent::new(
    //         0.0,
    //         learning_rate,
    //         discount_factor,
    //         initial_epsilon,
    //         epsilon_decay,
    //         final_epsilon,
    //         lambda_factor,
    //         sarsa,
    //     );

    // let mut trace_qlearning: ElegibilityTracesTabularEGreedyAgent<usize, SIZE> =
    //     ElegibilityTracesTabularEGreedyAgent::new(
    //         0.0,
    //         learning_rate,
    //         discount_factor,
    //         initial_epsilon,
    //         epsilon_decay,
    //         final_epsilon,
    //         lambda_factor,
    //         qlearning,
    //     );

    // let mut trace_expected_sarsa: ElegibilityTracesTabularEGreedyAgent<usize, SIZE> =
    //     ElegibilityTracesTabularEGreedyAgent::new(
    //         0.0,
    //         learning_rate,
    //         discount_factor,
    //         initial_epsilon,
    //         epsilon_decay,
    //         final_epsilon,
    //         lambda_factor,
    //         expected_sarsa,
    //     );

    let mut ucb_sarsa: OneStepTabularAgent<usize, SIZE> =
    OneStepTabularAgent::new(
        0.0,
        learning_rate,
        discount_factor,
        &mut aucb,
        sarsa,
    );

    let mut ucb_qlearning: OneStepTabularAgent<usize, SIZE> =
    OneStepTabularAgent::new(
        0.0,
        learning_rate,
        discount_factor,
        &mut bucb,
        qlearning,
    );

    let mut ucb_expected_sarsa: OneStepTabularAgent<usize, SIZE> =
    OneStepTabularAgent::new(
        0.0,
        learning_rate,
        discount_factor,
        &mut cucb,
        expected_sarsa,
    );

    let mut agents: Vec<&mut dyn Agent<usize, SIZE>> = vec![];
    agents.push(&mut step_sarsa);
    agents.push(&mut step_qlearning);
    agents.push(&mut step_expected_sarsa);
    // agents.push(&mut trace_sarsa);
    // agents.push(&mut trace_qlearning);
    // agents.push(&mut trace_expected_sarsa);
    agents.push(&mut ucb_sarsa);
    agents.push(&mut ucb_qlearning);
    agents.push(&mut ucb_expected_sarsa);

    for (i, agent) in agents.into_iter().enumerate() {
        let now: Instant = Instant::now();
        let (reward_history, episode_length) = agent.train(&mut env, n_episodes);
        let elapsed: std::time::Duration = now.elapsed();
        println!("{} {:.2?}", legends[i], elapsed);

        let ma_error = moving_average(
            agent.get_training_error().len() as usize / moving_average_window,
            &agent.get_training_error(),
        );
        errors.push(ma_error);
        let ma_reward =
            moving_average(n_episodes as usize / moving_average_window, &reward_history);
        rewards.push(ma_reward);
        let ma_episode = moving_average(
            n_episodes as usize / moving_average_window,
            &episode_length.iter().map(|x| *x as f64).collect(),
        );
        episodes_length.push(ma_episode);

        if cli.show_example {
            agent.example(&mut env);
        }
    }

    plot_moving_average(&rewards, &colors, &legends, "Rewards");

    plot_moving_average(&episodes_length, &colors, &legends, "Episodes Length");

    plot_moving_average(&errors, &colors, &legends, "Training Error");
}
