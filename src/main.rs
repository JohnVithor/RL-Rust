use std::cell::RefCell;
use std::time::Instant;

use plotters::style::{BLUE, GREEN, RED, YELLOW};

use reinforcement_learning::algorithms::action_selection::{UniformEpsilonGreed, UpperConfidenceBound, self};
use reinforcement_learning::algorithms::policy_update::{QStep, SarsaStep, SarsaLambda, QLambda};
use reinforcement_learning::env::{Env, BlackJackEnv, FrozenLakeEnv, CliffWalkingEnv, TaxiEnv};
use reinforcement_learning::utils::{moving_average, plot_moving_average};
use reinforcement_learning::{Agent, Policy};

extern crate structopt;

use structopt::StructOpt;

/// Train four RL agents using some parameters and generate some graphics of their results
#[derive(StructOpt, Debug)]
#[structopt(name = "RLRust")]
struct Cli {

    /// Name of the environment to be used: 0: blackjack, 1: frozenlake, 2: cliffwalk, 3: taxi
    env: u8,

    /// Should the env be stochastic
    #[structopt(long = "stochastic_env", short = "se")]
    stochastic_env: bool,

    /// Number of episodes for the training
    #[structopt(long = "n_episodes", short = "epi", default_value = "100000")]
    n_episodes: u128,

    /// Maximum number of steps per episode
    #[structopt(long = "max_steps", short = "ms", default_value = "100")]
    max_steps: u128,

    /// Learning rate of the RL agent
    #[structopt(long = "learning_rate", short = "lr", default_value = "0.05")]
    learning_rate: f64,

    /// Action selection strategy: 0 - uniform epsilon greed, 1 - upper confidence bound
    #[structopt(long = "action_selection", short = "as", default_value = "0")]
    action_selection: u8,

    /// Initial value for the exploration ratio
    #[structopt(long = "initial_epsilon", short = "ie", default_value = "1.0")]
    initial_epsilon: f64,

    /// Value to decrease of the exploration ratio at each step default to: initial_epsilon / (n_episodes / 2);
    #[structopt(long = "epsilon_decay", short = "ed", default_value = "NAN")]
    epsilon_decay: f64,

    /// Final value for the exploration ratio
    #[structopt(long = "final_epsilon", short = "fe", default_value = "0.0")]
    final_epsilon: f64,

    /// Confidence level for the UCB action selection strategy
    #[structopt(long = "confidence_level", short = "cl", default_value = "0.5")]
    confidence_level: f64,

    /// Discont factor to be used on the temporal difference calculation
    #[structopt(long = "discount_factor", short = "df", default_value = "0.95")]
    discount_factor: f64,

    /// Lambda factor to be used on the eligibility traces algorithms
    #[structopt(long = "lambda_factor", short = "lf", default_value = "0.5")]
    lambda_factor: f64,

    /// Moving average window to be used on the visualization of results
    #[structopt(long = "moving_average_window", short = "maw", default_value = "100")]
    moving_average_window: usize,
}

fn main() {
    
    let cli: Cli = Cli::from_args();

    let n_episodes: u128 = cli.n_episodes;
    let max_steps: u128 = cli.max_steps;

    let learning_rate: f64 = cli.learning_rate;
    let initial_epsilon: f64 = cli.initial_epsilon;
    let epsilon_decay: f64 = if cli.epsilon_decay.is_nan() {initial_epsilon / (n_episodes as f64 / 2.0)} else {cli.epsilon_decay};
    let final_epsilon: f64 = cli.final_epsilon;
    let confidence_level: f64 = cli.confidence_level;
    let discount_factor: f64 = cli.discount_factor;
    let lambda_factor: f64 = cli.lambda_factor;
    
    let moving_average_window: usize = cli.moving_average_window;

    let mut blackjack = BlackJackEnv::new();
    let mut frozen_lake = FrozenLakeEnv::new(&FrozenLakeEnv::MAP_4X4, cli.stochastic_env, max_steps);
    let mut cliffwalking = CliffWalkingEnv::new(max_steps);
    let mut taxi = TaxiEnv::new(max_steps);

    let env: &mut dyn Env<usize> = match cli.env {
        0 =>  {println!("Selected env blackjack"); &mut blackjack},
        1 =>  {println!("Selected env frozen_lake"); &mut frozen_lake},
        2 =>  {println!("Selected env cliffwalking"); &mut cliffwalking},
        3 =>  {println!("Selected env taxi"); &mut taxi},
        _ =>  panic!("Not a valid env selected")
    };

    // println!("Play!");
    // env.play();

    let mut rewards: Vec<&Vec<f64>> = vec![];
    let mut episodes_length: Vec<&Vec<f64>> = vec![];
    let mut errors : Vec<&Vec<f64>> = vec![];

    let mut action_selection_strategy = UniformEpsilonGreed::new(initial_epsilon, epsilon_decay, final_epsilon);
    let mut policy_update_strategy: SarsaStep = SarsaStep::new(learning_rate, discount_factor);
    let epsilon_greed_sarsa = &mut Agent::new(
        &mut action_selection_strategy,
        &mut policy_update_strategy,
        Policy::new(0.0, env.action_space()),
        env.action_space(),
    );

    let mut action_selection_strategy = UniformEpsilonGreed::new(initial_epsilon, epsilon_decay, final_epsilon);
    let mut policy_update_strategy = QStep::new(learning_rate, discount_factor);
    let epsilon_greed_qlearning = &mut Agent::new(
        &mut action_selection_strategy,
        &mut policy_update_strategy,
        Policy::new(0.0, env.action_space()),
        env.action_space(),
    );

    let mut action_selection_strategy = UniformEpsilonGreed::new(initial_epsilon, epsilon_decay, final_epsilon);
    let mut policy_update_strategy = SarsaLambda::new(learning_rate, discount_factor, lambda_factor, 0.0, env.action_space());
    let epsilon_greed_sarsa_lambda = &mut Agent::new(
        &mut action_selection_strategy,
        &mut policy_update_strategy,
        Policy::new(0.0, env.action_space()),
        env.action_space(),
    );

    let mut action_selection_strategy = UniformEpsilonGreed::new(initial_epsilon, epsilon_decay, final_epsilon);
    let mut policy_update_strategy = QLambda::new(learning_rate, discount_factor, lambda_factor, 0.0, env.action_space());
    let epsilon_greed_qlearning_lambda = &mut Agent::new(
        &mut action_selection_strategy,
        &mut policy_update_strategy,
        Policy::new(0.0, env.action_space()),
        env.action_space(),
    );

    let mut action_selection_strategy = UpperConfidenceBound::new(confidence_level);
    let mut policy_update_strategy = SarsaStep::new(learning_rate, discount_factor);
    let ucb_sarsa = &mut Agent::new(
        &mut action_selection_strategy,
        &mut policy_update_strategy,
        Policy::new(0.0, env.action_space()),
        env.action_space(),
    );

    let mut action_selection_strategy = UpperConfidenceBound::new(confidence_level);
    let mut policy_update_strategy = QStep::new(learning_rate, discount_factor);
    let ucb_qlearning = &mut Agent::new(
        &mut action_selection_strategy,
        &mut policy_update_strategy,
        Policy::new(0.0, env.action_space()),
        env.action_space(),
    );

    let mut action_selection_strategy = UpperConfidenceBound::new(confidence_level);
    let mut policy_update_strategy = SarsaLambda::new(learning_rate, discount_factor, lambda_factor, 0.0, env.action_space());
    let ucb_sarsa_lambda = &mut Agent::new(
        &mut action_selection_strategy,
        &mut policy_update_strategy,
        Policy::new(0.0, env.action_space()),
        env.action_space(),
    );

    let mut action_selection_strategy = UpperConfidenceBound::new(confidence_level);
    let mut policy_update_strategy = QLambda::new(learning_rate, discount_factor, lambda_factor, 0.0, env.action_space());
    let ucb_qlearning_lambda = &mut Agent::new(
        &mut action_selection_strategy,
        &mut policy_update_strategy,
        Policy::new(0.0, env.action_space()),
        env.action_space(),
    );

    let sarsa = match cli.action_selection {
        0 => epsilon_greed_sarsa,
        1 => ucb_sarsa,
        _ => panic!("Action selection invalid!")
    };

    let qlearning = match cli.action_selection {
        0 => epsilon_greed_qlearning,
        1 => ucb_qlearning,
        _ => panic!("Action selection invalid!")
    };

    let sarsa_lambda = match cli.action_selection {
        0 => epsilon_greed_sarsa_lambda,
        1 => ucb_sarsa_lambda,
        _ => panic!("Action selection invalid!")
    };

    let qlearning_lambda = match cli.action_selection {
        0 => epsilon_greed_qlearning_lambda,
        1 => ucb_qlearning_lambda,
        _ => panic!("Action selection invalid!")
    };

    let now: Instant = Instant::now();
    let (reward_history, episode_length)  = sarsa.train(env, n_episodes);
    let elapsed: std::time::Duration = now.elapsed();
    println!("Sarsa {:.2?}", elapsed);

    let ma_error = moving_average(sarsa.get_training_error().len() as usize / moving_average_window, &sarsa.get_training_error());
    errors.push(&ma_error);
    let ma_reward = moving_average(n_episodes as usize / moving_average_window, &reward_history);
    rewards.push(&ma_reward);
    let ma_episode = moving_average(n_episodes as usize / moving_average_window, &episode_length.iter().map(|x| *x as f64).collect());
    episodes_length.push(&ma_episode);

    let now: Instant = Instant::now();
    let (reward_history, episode_length)  = qlearning.train(env, n_episodes);
    let elapsed: std::time::Duration = now.elapsed();
    println!("QLearning {:.2?}", elapsed);

    let ma_error = moving_average(qlearning.get_training_error().len() as usize / moving_average_window, &qlearning.get_training_error());
    errors.push(&ma_error);
    let ma_reward = moving_average(n_episodes as usize / moving_average_window, &reward_history);
    rewards.push(&ma_reward);
    let ma_episode = moving_average(n_episodes as usize / moving_average_window, &episode_length.iter().map(|x| *x as f64).collect());
    episodes_length.push(&ma_episode);

    let now: Instant = Instant::now();
    let (reward_history, episode_length)  = sarsa_lambda.train(env, n_episodes);
    let elapsed: std::time::Duration = now.elapsed();
    println!("SarsaLambda {:.2?}", elapsed);

    let ma_error = moving_average(sarsa_lambda.get_training_error().len() as usize / moving_average_window, &sarsa_lambda.get_training_error());
    errors.push(&ma_error);
    let ma_reward = moving_average(n_episodes as usize / moving_average_window, &reward_history);
    rewards.push(&ma_reward);
    let ma_episode = moving_average(n_episodes as usize / moving_average_window, &episode_length.iter().map(|x| *x as f64).collect());
    episodes_length.push(&ma_episode);

    let now: Instant = Instant::now();
    let (reward_history, episode_length)  = qlearning_lambda.train(env, n_episodes);
    let elapsed: std::time::Duration = now.elapsed();
    println!("QLearningLambda {:.2?}", elapsed);

    let legends: Vec<&str> = [
        "Sarsa",
        "Qlearning",
        "SarsaLambda",
        "QlearningLambda"
    ].to_vec();

    let colors: Vec<&plotters::style::RGBColor> = [&BLUE, &GREEN, &RED, &YELLOW].to_vec();

    let ma_error = moving_average(qlearning_lambda.get_training_error().len() as usize / moving_average_window, &qlearning_lambda.get_training_error());
    errors.push(&ma_error);
    let ma_reward = moving_average(n_episodes as usize / moving_average_window, &reward_history);
    rewards.push(&ma_reward);
    let ma_episode = moving_average(n_episodes as usize / moving_average_window, &episode_length.iter().map(|x| *x as f64).collect());
    episodes_length.push(&ma_episode);

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
