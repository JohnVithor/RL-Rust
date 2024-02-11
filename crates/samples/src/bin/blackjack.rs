use std::time::Instant;

extern crate environments;
extern crate reinforcement_learning;
extern crate structopt;

use environments::toy_text::blackjack::{BlackJackAction, BlackJackEnv, BlackJackObservation};
use environments::DiscreteEnv;
use reinforcement_learning::trainer::DiscreteTrainer;
use serde_json::json;
use structopt::StructOpt;

use samples::{get_agents, moving_average, save_json, Cli};

fn main() {
    let cli: Cli = Cli::from_args();

    let n_episodes: u128 = cli.n_episodes;

    let moving_average_window: usize = cli.moving_average_window;

    let mut env = BlackJackEnv::new();

    let (agents, identifiers) = get_agents(cli);

    let mut train_rewards: Vec<Vec<f64>> = vec![];
    let mut train_episodes_length: Vec<Vec<f64>> = vec![];
    let mut train_errors: Vec<Vec<f64>> = vec![];
    let mut test_rewards: Vec<Vec<f64>> = vec![];
    let mut test_episodes_length: Vec<Vec<f64>> = vec![];

    for (i, mut agent) in agents.into_iter().enumerate() {
        println!("{} has:", identifiers[i]);
        let mut trainer = DiscreteTrainer::new(
            |repr: usize| -> BlackJackAction {
                match repr {
                    0 => BlackJackAction::HIT,
                    1 => BlackJackAction::STICK,
                    repr => panic!(
                        "Invalid value to convert from usize to BlackJackAction: {}",
                        repr
                    ),
                }
            },
            |repr: &BlackJackObservation| -> usize {
                let p_score = repr.p_score as usize;
                let d_score = repr.d_score as usize;
                let p_ace = repr.p_ace as usize;
                56 * (p_score - 4) + 2 * (d_score - 2) + p_ace
            },
        );
        let now: Instant = Instant::now();
        let (
            training_reward,
            training_length,
            training_error,
            _evaluation_reward,
            _evaluation_length,
        ) = trainer.train(
            &mut env,
            agent.as_mut(),
            n_episodes,
            n_episodes / 10,
            100,
            false,
        );
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

        let mut wins: u32 = 0;
        let mut losses: u32 = 0;
        let mut draws: u32 = 0;
        const LOOP_LEN: usize = 1000000;
        for _u in 0..LOOP_LEN {
            let mut curr_action: BlackJackAction =
                (trainer.repr_to_action)(agent.get_action((trainer.obs_to_repr)(&env.reset())));
            loop {
                let (next_obs, reward, terminated) = env.step(curr_action).unwrap();
                let next_action: BlackJackAction =
                    (trainer.repr_to_action)(agent.get_action((trainer.obs_to_repr)(&next_obs)));
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
            " - win-rate of {:.2?}%\n - loss-rate of {:.2?}%\n - draw-rate {:.2?}%",
            wins as f64 / LOOP_LEN as f64,
            losses as f64 / LOOP_LEN as f64,
            draws as f64 / LOOP_LEN as f64
        );

        let (testing_rewards, testing_length) =
            trainer.evaluate(&mut env, agent.as_mut(), n_episodes);

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
