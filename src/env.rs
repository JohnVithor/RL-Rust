use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::rc::Rc;

mod action_space;
mod blackjack;
mod frozen_lake;
mod cliff_walking;
mod taxi;

pub use action_space::ActionSpace;
pub use blackjack::BlackJackEnv;
pub use frozen_lake::FrozenLakeEnv;
pub use cliff_walking::CliffWalkingEnv;
pub use taxi::TaxiEnv;

use crate::observation::Observation;
#[derive(Debug, Clone)]
pub struct EnvNotReady;

pub trait Env {
    fn reset(&mut self) -> Rc<dyn Observation>;
    fn step(&mut self, action: usize) -> Result<(Rc<dyn Observation>, f64, bool), EnvNotReady>;
    fn action_space(&self) -> ActionSpace;
    fn play(&mut self) {
        use std::io;
        let mut curr_obs: Rc<dyn Observation> = self.reset();
        let mut final_reward: f64 = 0.0;
        loop {
            println!("curr_obs {:?}", curr_obs);
            let mut user_input: String = String::new();
            io::stdin().read_line(&mut user_input).unwrap();
            user_input = user_input.trim().to_string();
            let curr_action: usize = user_input.parse::<usize>().unwrap();
            println!("selected_action {:?}", curr_action);
            let (next_obs, reward, terminated) = self.step(curr_action).unwrap();
            println!("reward {:?}", reward);
            final_reward += reward;
            curr_obs = next_obs;
            if terminated {
                println!("final_obs {:?}", curr_obs);
                println!("final_reward {:?}", final_reward);
                break;
            }
        }
    }
    fn render(&self) -> String;
    fn get_action_label(&self, action: usize) -> &str;
}