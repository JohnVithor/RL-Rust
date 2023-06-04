use std::time::Instant;

use plotters::style::{BLUE, GREEN, RED, YELLOW, CYAN, MAGENTA};

use reinforcement_learning::algorithms::action_selection::{UniformEpsilonGreed, UpperConfidenceBound};
use reinforcement_learning::algorithms::policy_update::{NextQvalueFunction, sarsa, q_learning, expected_sarsa};
use reinforcement_learning::env::{Env, BlackJackEnv, FrozenLakeEnv, CliffWalkingEnv, TaxiEnv};
use reinforcement_learning::policy::{DoublePolicy, TracePolicy, TraceMode, Policy};
use reinforcement_learning::utils::{moving_average, plot_moving_average};
use reinforcement_learning::{Agent, policy::BasicPolicy};

extern crate structopt;

use structopt::StructOpt;

/// Train four RL agents using some parameters and generate some graphics of their results
#[derive(StructOpt, Debug)]
#[structopt(name = "RLRust")]
struct Cli {

    /// Name of the environment to be used: 0: blackjack, 1: frozenlake, 2: cliffwalk, 3: taxi
    env: u8,

    /// Should the env be stochastic
    #[structopt(long = "stochastic_env")]
    stochastic_env: bool,

    /// Should the env be stochastic
    #[structopt(long = "show_example")]
    show_example: bool,

    /// Change the env's map, if possible
    #[structopt(long = "map", default_value = "4x4")]
    map: String,

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
    let map: String = cli.map;

    let learning_rate: f64 = cli.learning_rate;
    let initial_epsilon: f64 = cli.initial_epsilon;
    let epsilon_decay: f64 = if cli.epsilon_decay.is_nan() {initial_epsilon / (9.0 * n_episodes as f64 / 10.0)} else {cli.epsilon_decay};
    let final_epsilon: f64 = cli.final_epsilon;
    let confidence_level: f64 = cli.confidence_level;
    let discount_factor: f64 = cli.discount_factor;
    let lambda_factor: f64 = cli.lambda_factor;
    
    let moving_average_window: usize = cli.moving_average_window;

    let mut blackjack = BlackJackEnv::new();
    let mut frozen_lake;
    if cli.env == 1 && map == "8X8" {
        frozen_lake = FrozenLakeEnv::new(&FrozenLakeEnv::MAP_8X8, cli.stochastic_env, max_steps);
    } else if cli.env == 1 && map == "4x4" {
        frozen_lake = FrozenLakeEnv::new(&FrozenLakeEnv::MAP_4X4, cli.stochastic_env, max_steps);
    } else if cli.env != 1 {
        frozen_lake = FrozenLakeEnv::new(&FrozenLakeEnv::MAP_4X4, cli.stochastic_env, max_steps);
    } else {
        panic!("Map option of {:?} is not supported!", map);
    }
    let mut cliffwalking = CliffWalkingEnv::new(max_steps);
    let mut taxi = TaxiEnv::new(max_steps);

    let env: &mut dyn Env<usize> = match cli.env {
        0 =>  {println!("Selected env blackjack"); &mut blackjack},
        1 =>  {println!("Selected env frozen_lake{}", &map); &mut frozen_lake},
        2 =>  {println!("Selected env cliffwalking"); &mut cliffwalking},
        3 =>  {println!("Selected env taxi"); &mut taxi},
        _ =>  panic!("Not a valid env selected")
    };

    // println!("Play!");
    // env.play();

    let mut rewards: Vec<Vec<f64>> = vec![];
    let mut episodes_length: Vec<Vec<f64>> = vec![];
    let mut errors : Vec<Vec<f64>> = vec![];

    let mut basic_policy = BasicPolicy::new(learning_rate, discount_factor, 0.0, env.action_space());
    let mut double_policy = DoublePolicy::new(learning_rate, discount_factor, 0.0, env.action_space());

    let mut sarsa_trace_policy = TracePolicy::new(Box::new(basic_policy.clone()), lambda_factor, TraceMode::SARSA);
    let mut qlearning_trace_policy = TracePolicy::new(Box::new(basic_policy.clone()), lambda_factor, TraceMode::SARSA);

    let mut double_sarsa_trace_policy = TracePolicy::new(Box::new(double_policy.clone()), lambda_factor, TraceMode::SARSA);
    let mut double_qlearning_trace_policy = TracePolicy::new(Box::new(double_policy.clone()), lambda_factor, TraceMode::SARSA);

    let mut uniforme_psilon_greed = UniformEpsilonGreed::new(initial_epsilon, epsilon_decay, final_epsilon);
    let mut ucb = UpperConfidenceBound::new(confidence_level);
    
    let legends: Vec<&str> = [
        "Sarsa",
        "Qlearning",
        "One-Step Expected Sarsa",
        "Sarsa Lambda",
        "Qlearning Lambda",
    ].to_vec();

    let colors: Vec<&plotters::style::RGBColor> = [
        &BLUE,
        &GREEN,
        &CYAN,
        &RED,
        &YELLOW
    ].to_vec();

    let mut policy_update_strategies: Vec<(&str, NextQvalueFunction<usize>, &mut dyn Policy<usize>)> = vec![];
    let mut a = basic_policy.clone();
    let mut b = basic_policy.clone();   
    let mut c = double_policy.clone();
    let mut d = double_policy.clone();   
    if cli.policy_type == 0 {
        policy_update_strategies.push((legends[0], sarsa, &mut basic_policy));
        policy_update_strategies.push((legends[1], q_learning, &mut a));
        policy_update_strategies.push((legends[2], expected_sarsa, &mut b));
        policy_update_strategies.push((legends[3], sarsa, &mut sarsa_trace_policy));
        policy_update_strategies.push((legends[4], q_learning, &mut qlearning_trace_policy));
    } else {
        policy_update_strategies.push((legends[0], sarsa, &mut double_policy));
        policy_update_strategies.push((legends[1], q_learning, &mut c));
        policy_update_strategies.push((legends[2], expected_sarsa, &mut d));
        policy_update_strategies.push((legends[3], sarsa, &mut double_sarsa_trace_policy));
        policy_update_strategies.push((legends[4], q_learning, &mut double_qlearning_trace_policy));
    }

    for (n,s, p) in policy_update_strategies {
        let agent = &mut Agent::new(
            if cli.action_selection == 0 {&mut uniforme_psilon_greed} else {&mut ucb},
            s,
            p,
            env.action_space(),
        );
        let now: Instant = Instant::now();
        let (reward_history, episode_length)  = agent.train(env, n_episodes);
        let elapsed: std::time::Duration = now.elapsed();
        println!("{} {:.2?}", n.clone(), elapsed);

        let ma_error = moving_average(agent.get_training_error().len() as usize / moving_average_window, &agent.get_training_error());
        errors.push(ma_error);
        let ma_reward = moving_average(n_episodes as usize / moving_average_window, &reward_history);
        rewards.push(ma_reward);
        let ma_episode = moving_average(n_episodes as usize / moving_average_window, &episode_length.iter().map(|x| *x as f64).collect());
        episodes_length.push(ma_episode);
        if cli.show_example {
            agent.example(env);
        }
    }

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
