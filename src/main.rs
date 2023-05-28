use std::cell::RefCell;
use std::time::Instant;

use plotters::style::{BLUE, WHITE, GREEN, RED, YELLOW};

use reinforcement_learning::algorithms::action_selection::{UniformEpsilonGreed, UpperConfidenceBound};
use reinforcement_learning::algorithms::policy_update::{QStep, SarsaStep, SarsaLambda, QLambda};
use reinforcement_learning::env::{Env, EnvNotReady, BlackJackEnv, FrozenLakeEnv, CliffWalkingEnv, TaxiEnv};
use reinforcement_learning::utils::{moving_average, plot_moving_average};
use reinforcement_learning::{Agent, Policy};

fn main() {
    let learning_rate: f64 = 0.05;
    let n_episodes: u128 = 100_000;
    let moving_average_window: usize = 50;
    let start_epsilon: f64 = 1.0;
    let epsilon_decay: f64 = start_epsilon / (n_episodes as f64 / 2.0);
    let final_epsilon: f64 = 0.0;
    let confidence_level: f64 = 0.8;
    let discount_factor: f64 = 0.95;
    let lambda_factor: f64 = 0.5;
    let max_steps: u128 = 100;
    // let mut env = BlackJackEnv::new();
    let mut env = FrozenLakeEnv::new(&FrozenLakeEnv::MAP_4X4, false, max_steps);
    // let mut env = CliffWalkingEnv::new(max_steps);
    // let mut env = TaxiEnv::new(max_steps);

    println!("Play!");
    // env.play();

    // let policy_update_strategy = QLambda::new(learning_rate, discount_factor, lambda_factor, 0.0, env.action_space());
    // let policy_update_strategy = QStep::new(learning_rate, discount_factor);
    // let action_selection_strategy = UniformEpsilonGreed::new(start_epsilon, epsilon_decay, final_epsilon);
    // let action_selection_strategy = UpperConfidenceBound::new(confidence_level);
    // let policy = Policy::new(0.0, env.action_space());

    let mut rewards: Vec<&Vec<f64>> = vec![];
    let mut episodes_length: Vec<&Vec<f64>> = vec![];
    let mut errors : Vec<&Vec<f64>> = vec![];

    let epsilon_greed_sarsa = &mut Agent::new(
        Box::new(RefCell::new(UniformEpsilonGreed::new(start_epsilon, epsilon_decay, final_epsilon))),
        Box::new(RefCell::new(SarsaStep::new(learning_rate, discount_factor))),
        Policy::new(0.0, env.action_space()),
        env.action_space(),
    );

    let epsilon_greed_qlearning = &mut Agent::new(
        Box::new(RefCell::new(UniformEpsilonGreed::new(start_epsilon, epsilon_decay, final_epsilon))),
        Box::new(RefCell::new(QStep::new(learning_rate, discount_factor))),
        Policy::new(0.0, env.action_space()),
        env.action_space(),
    );

    let epsilon_greed_sarsa_lambda = &mut Agent::new(
        Box::new(RefCell::new(UniformEpsilonGreed::new(start_epsilon, epsilon_decay, final_epsilon))),
        Box::new(RefCell::new(SarsaLambda::new(learning_rate, discount_factor, lambda_factor, 0.0, env.action_space()))),
        Policy::new(0.0, env.action_space()),
        env.action_space(),
    );

    let epsilon_greed_qlearning_lambda = &mut Agent::new(
        Box::new(RefCell::new(UniformEpsilonGreed::new(start_epsilon, epsilon_decay, final_epsilon))),
        Box::new(RefCell::new(QLambda::new(learning_rate, discount_factor, lambda_factor, 0.0, env.action_space()))),
        Policy::new(0.0, env.action_space()),
        env.action_space(),
    );

    println!("Training EpsilonGreedSarsa agent!");
    let now: Instant = Instant::now();
    let (reward_history, episode_length)  = epsilon_greed_sarsa.train(&mut env, n_episodes);
    let elapsed: std::time::Duration = now.elapsed();
    println!("Training done!");
    println!("Time elapsed: {:.2?}", elapsed);

    let ma_error = moving_average(epsilon_greed_sarsa.get_training_error().len() as usize / moving_average_window, &epsilon_greed_sarsa.get_training_error());
    errors.push(&ma_error);
    let ma_reward = moving_average(n_episodes as usize / moving_average_window, &reward_history);
    rewards.push(&ma_reward);
    let ma_episode = moving_average(n_episodes as usize / moving_average_window, &episode_length.iter().map(|x| *x as f64).collect());
    episodes_length.push(&ma_episode);

    println!("Training EpsilonGreedQLearning agent!");
    let now: Instant = Instant::now();
    let (reward_history, episode_length)  = epsilon_greed_qlearning.train(&mut env, n_episodes);
    let elapsed: std::time::Duration = now.elapsed();
    println!("Training done!");
    println!("Time elapsed: {:.2?}", elapsed);

    let ma_error = moving_average(epsilon_greed_qlearning.get_training_error().len() as usize / moving_average_window, &epsilon_greed_sarsa.get_training_error());
    errors.push(&ma_error);
    let ma_reward = moving_average(n_episodes as usize / moving_average_window, &reward_history);
    rewards.push(&ma_reward);
    let ma_episode = moving_average(n_episodes as usize / moving_average_window, &episode_length.iter().map(|x| *x as f64).collect());
    episodes_length.push(&ma_episode);

    println!("Training EpsilonGreedSarsaLambda agent!");
    let now: Instant = Instant::now();
    let (reward_history, episode_length)  = epsilon_greed_sarsa_lambda.train(&mut env, n_episodes);
    let elapsed: std::time::Duration = now.elapsed();
    println!("Training done!");
    println!("Time elapsed: {:.2?}", elapsed);

    let ma_error = moving_average(epsilon_greed_sarsa_lambda.get_training_error().len() as usize / moving_average_window, &epsilon_greed_sarsa.get_training_error());
    errors.push(&ma_error);
    let ma_reward = moving_average(n_episodes as usize / moving_average_window, &reward_history);
    rewards.push(&ma_reward);
    let ma_episode = moving_average(n_episodes as usize / moving_average_window, &episode_length.iter().map(|x| *x as f64).collect());
    episodes_length.push(&ma_episode);

    println!("Training EpsilonGreedQLearningLambda agent!");
    let now: Instant = Instant::now();
    let (reward_history, episode_length)  = epsilon_greed_qlearning_lambda.train(&mut env, n_episodes);
    let elapsed: std::time::Duration = now.elapsed();
    println!("Training done!");
    println!("Time elapsed: {:.2?}", elapsed);

    let ma_error = moving_average(epsilon_greed_qlearning_lambda.get_training_error().len() as usize / moving_average_window, &epsilon_greed_sarsa.get_training_error());
    errors.push(&ma_error);
    let ma_reward = moving_average(n_episodes as usize / moving_average_window, &reward_history);
    rewards.push(&ma_reward);
    let ma_episode = moving_average(n_episodes as usize / moving_average_window, &episode_length.iter().map(|x| *x as f64).collect());
    episodes_length.push(&ma_episode);

    plot_moving_average(
        &rewards,
        &[&BLUE, &GREEN, &RED, &YELLOW].to_vec(),
        &[
            "EpsilonGreedSarsaBaseline",
            "EpsilonGreedQlearningBaseline",
            "EpsilonGreedSarsaLambdaBaseline",
            "EpsilonGreedQlearningLambdaBaseline"
        ].to_vec(),
        "Rewards"
    );

    plot_moving_average(
        &episodes_length,
        &[&BLUE, &GREEN, &RED, &YELLOW].to_vec(),
        &[
            "EpsilonGreedSarsaBaseline",
            "EpsilonGreedQlearningBaseline",
            "EpsilonGreedSarsaLambdaBaseline",
            "EpsilonGreedQlearningLambdaBaseline"
        ].to_vec(),
        "Episodes Length"
    );

    plot_moving_average(
        &errors,
        &[&BLUE, &GREEN, &RED, &YELLOW].to_vec(),
        &[
            "EpsilonGreedSarsaBaseline",
            "EpsilonGreedQlearningBaseline",
            "EpsilonGreedSarsaLambdaBaseline",
            "EpsilonGreedQlearningLambdaBaseline"
        ].to_vec(),
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
