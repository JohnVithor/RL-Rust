use std::time::Instant;

extern crate environments;
extern crate reinforcement_learning;
extern crate structopt;

use environments::toy_text::frozen_lake::{FrozenLakeAction, FrozenLakeEnv};
use reinforcement_learning::trainer::FullDiscreteTrainer;
use samples::{get_agents, moving_average, save_json, Cli};
use serde_json::json;
use structopt::StructOpt;

fn main() {
    let cli: Cli = Cli::from_args();

    let n_episodes: u128 = cli.n_episodes;
    let max_steps: u128 = cli.max_steps;
    let moving_average_window: usize = cli.moving_average_window;

    let mut env = FrozenLakeEnv::new(
        if cli.map == "4x4" {
            &FrozenLakeEnv::MAP_4X4
        } else if cli.map == "8x8" {
            &FrozenLakeEnv::MAP_8X8
        } else {
            panic!("Invalid map: {}", cli.map)
        },
        cli.stochastic_env,
        max_steps,
        cli.seed,
    );

    let mut train_rewards: Vec<Vec<f32>> = vec![];
    let mut train_episodes_length: Vec<Vec<f32>> = vec![];
    let mut train_errors: Vec<Vec<f32>> = vec![];
    let mut test_rewards: Vec<Vec<f32>> = vec![];
    let mut test_episodes_length: Vec<Vec<f32>> = vec![];

    let (agents, identifiers) = get_agents(cli);

    for (i, mut agent) in agents.into_iter().enumerate() {
        println!("{} has:", identifiers[i]);
        let mut trainer = FullDiscreteTrainer::new(
            |repr: usize| -> FrozenLakeAction {
                match repr {
                    0 => FrozenLakeAction::LEFT,
                    1 => FrozenLakeAction::DOWN,
                    2 => FrozenLakeAction::RIGHT,
                    3 => FrozenLakeAction::UP,
                    repr => panic!(
                        "Invalid value to convert from usize to FrozenLakeAction: {}",
                        repr
                    ),
                }
            },
            |repr: &usize| -> usize { *repr },
        );
        let now: Instant = Instant::now();
        let (
            training_reward,
            training_length,
            training_error,
            _evaluation_reward,
            _evaluation_length,
        ) = trainer.train_by_episode(
            &mut env,
            agent.as_mut(),
            n_episodes,
            n_episodes / 10,
            100,
            false,
        );
        let elapsed: std::time::Duration = now.elapsed();
        println!(" - training time of {:.2?}", elapsed);

        // trainer.example(&mut env, agent);

        let ma_error = moving_average(
            training_error.len() / moving_average_window,
            &training_error,
        );
        train_errors.push(ma_error);
        let ma_reward = moving_average(
            n_episodes as usize / moving_average_window,
            &training_reward,
        );
        let v: Vec<f32> = training_length.iter().map(|x| *x as f32).collect();
        train_rewards.push(ma_reward);
        let ma_episode = moving_average(n_episodes as usize / moving_average_window, &v);
        train_episodes_length.push(ma_episode);

        let (testing_rewards, testing_length) =
            trainer.evaluate(&mut env, agent.as_mut(), n_episodes);

        let ma_reward = moving_average(
            n_episodes as usize / moving_average_window,
            &testing_rewards,
        );
        test_rewards.push(ma_reward);
        let v: Vec<f32> = testing_length.iter().map(|x| *x as f32).collect();
        let ma_episode = moving_average(n_episodes as usize / moving_average_window, &v);
        test_episodes_length.push(ma_episode);

        let mut mean_reward = 0.0;
        for v in testing_rewards {
            mean_reward += v;
        }
        mean_reward /= n_episodes as f32;
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
