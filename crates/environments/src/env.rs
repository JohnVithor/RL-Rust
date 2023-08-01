use std::fmt::Debug;
use std::io;

#[derive(Debug, Clone)]
pub struct EnvNotReady;

pub trait Env<T: Debug, const COUNT: usize> {
    fn action_size(&self) -> usize {
        COUNT
    }
    fn reset(&mut self) -> T;
    fn step(&mut self, action: usize) -> Result<(T, f64, bool), EnvNotReady>;
    fn play(&mut self) {
        let mut curr_obs: T = self.reset();
        let mut final_reward: f64 = 0.0;
        loop {
            println!("curr_obs {:?}", curr_obs);
            println!("{}", self.render());
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
