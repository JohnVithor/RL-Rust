mod sarsa;

use std::hash::Hash;
use std::fmt::Debug;

pub use sarsa::OneStepTabularEGreedySarsa;

use crate::env::Env;


pub fn train<T: Hash+PartialEq+Eq+Clone+Debug, const COUNT: usize>(agent: &mut OneStepTabularEGreedySarsa<T, COUNT>, env: &mut dyn Env<T>, n_episodes: u128) -> (Vec<f64>, Vec<u128>) {
    let mut reward_history: Vec<f64> = vec![];
    let mut episode_length: Vec<u128> = vec![];
    for _episode in 0..n_episodes {
        let mut action_counter: u128 = 0;
        let mut epi_reward: f64 = 0.0;
        let mut curr_obs: T = env.reset();
        let mut curr_action: usize = agent.get_action(&curr_obs);
        loop {
            action_counter+=1;
            let (next_obs, reward, terminated) = env.step(curr_action).unwrap();
            let next_action: usize = agent.get_action(&next_obs);
            agent.update(
                &curr_obs,
                curr_action,
                reward,
                terminated,
                &next_obs,
                next_action,
            );
            curr_obs = next_obs;
            curr_action = next_action;
            epi_reward += reward;
            if terminated {
                reward_history.push(epi_reward);
                break;
            }
        }
        episode_length.push(action_counter);
    }
    return (reward_history, episode_length);
}

// pub fn example(&mut self, env: &mut dyn Env<T>) {
//     let mut epi_reward = 0.0;
//     let mut curr_action: usize = self.get_action(&env.reset());
//     let mut steps: i32 = 0;
//     for _i in 0..100 {
//         steps+=1;
//         println!("{}", env.render());
//         let (next_obs, reward, terminated) = env.step(curr_action).unwrap();
//         let next_action: usize = self.get_action(&next_obs);
//         println!("{:?}", env.get_action_label(curr_action));
//         println!("step reward {:?}", reward);
//         curr_action = next_action;
//         epi_reward+=reward;
//         if terminated {
//             println!("{}", env.render());
//             println!("episode reward {:?}", epi_reward);
//             println!("terminated with {:?} steps", steps);
//             break;
//         }
//     }
// }