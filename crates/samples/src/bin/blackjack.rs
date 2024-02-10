use std::fs;
use std::rc::Rc;
use std::time::Instant;

extern crate environments;
extern crate reinforcement_learning;
extern crate structopt;

use environments::toy_text::blackjack::{BlackJackAction, BlackJackEnv, BlackJackObservation};
use environments::DiscreteEnv;
use reinforcement_learning::action_selection::{UniformEpsilonGreed, UpperConfidenceBound};
use reinforcement_learning::agent::{expected_sarsa, qlearning, sarsa};
use reinforcement_learning::agent::{DiscreteAgent, ElegibilityTracesAgent, OneStepAgent};
use reinforcement_learning::policy::TabularPolicy;
use serde_json::json;
use structopt::StructOpt;

use samples::{moving_average, save_json};
/// Train RL agents using some parameters and export some data of their results
#[derive(StructOpt, Debug)]
#[structopt(name = "RLRust - BlackJack")]
struct Cli {
    /// Number of episodes for the training
    #[structopt(long = "n_episodes", short = "n", default_value = "100000")]
    n_episodes: u128,

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

    let learning_rate: f64 = cli.learning_rate;
    let initial_epsilon: f64 = cli.initial_epsilon;
    let epsilon_decay: f64 = initial_epsilon / (cli.exploration_time * n_episodes as f64);
    let final_epsilon: f64 = cli.final_epsilon;
    let confidence_level: f64 = cli.confidence_level;
    let discount_factor: f64 = cli.discount_factor;
    let lambda_factor: f64 = cli.lambda_factor;

    let moving_average_window: usize = cli.moving_average_window;

    let mut env = BlackJackEnv::new();

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

    let mut one_step_agent_epg: OneStepAgent<BlackJackObservation, BlackJackAction> =
        OneStepAgent::new(
            discount_factor,
            sarsa::<BlackJackAction>,
            &mut one_step_policy_epg,
            &mut one_step_epg,
        );
    let mut one_step_agent_ucb: OneStepAgent<BlackJackObservation, BlackJackAction> =
        OneStepAgent::new(
            discount_factor,
            sarsa::<BlackJackAction>,
            &mut one_step_policy_ucb,
            &mut one_step_ucb,
        );

    let mut trace_agent_epg: ElegibilityTracesAgent<BlackJackObservation, BlackJackAction> =
        ElegibilityTracesAgent::new(
            discount_factor,
            lambda_factor,
            sarsa::<BlackJackAction>,
            &mut trace_policy_epg,
            &mut trace_epg,
        );

    let mut trace_agent_ucb: ElegibilityTracesAgent<BlackJackObservation, BlackJackAction> =
        ElegibilityTracesAgent::new(
            discount_factor,
            lambda_factor,
            sarsa::<BlackJackAction>,
            &mut trace_policy_ucb,
            &mut trace_ucb,
        );

    let mut agents: Vec<&mut dyn DiscreteAgent<BlackJackObservation, BlackJackAction>> = vec![
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
            sarsa::<BlackJackAction>,
            qlearning::<BlackJackAction>,
            expected_sarsa::<BlackJackAction>,
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
            let v: Vec<f64> = training_length.iter().map(|x| *x as f64).collect();
            train_rewards.push(ma_reward);
            let ma_episode = moving_average(n_episodes as usize / moving_average_window, &v);
            train_episodes_length.push(ma_episode);

            let mut wins: u32 = 0;
            let mut losses: u32 = 0;
            let mut draws: u32 = 0;
            const LOOP_LEN: usize = 1000000;
            for _u in 0..LOOP_LEN {
                let mut curr_action: BlackJackAction = agent.get_action(&env.reset());
                loop {
                    let (next_obs, reward, terminated) = env.step(curr_action).unwrap();
                    let next_action: BlackJackAction = agent.get_action(&next_obs);
                    curr_action = next_action;
                    if terminated {
                        if reward == 1.0 {
                            wins += 1;
                        } else if reward == -1.0 {
                            losses += 1;
                        } else {
                            draws += 1;
                        }
                        break;
                    }
                }
            }
            println!(
                "{} has win-rate of {}%, loss-rate of {}% and draw-rate {}%",
                identifiers[i],
                wins as f64 / LOOP_LEN as f64,
                losses as f64 / LOOP_LEN as f64,
                draws as f64 / LOOP_LEN as f64
            );

            let (testing_rewards, testing_length) = agent.evaluate(&mut env, n_episodes);

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
            fs::write(
                &format!("{}_blackjackprobs.pair", identifiers[i]),
                &format!(
                    "{:?}, {:?}",
                    agent.get_policy().get_estimed_transitions(),
                    mean_reward
                ),
            )
            .expect("Unable to write file");
            // println!("{:?}", agent.get_policy().get_estimed_transitions());
            agent.reset();
            i += 1;
        }
    }
    println!("{:?}", one_step_policy_epg.state_changes_counter);
    println!("{:?}", one_step_policy_epg.state_action_change_counter);
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
// fn main() {}
