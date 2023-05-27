mod action_space;
mod blackjack;
mod frozen_lake;


#[derive(Debug, Clone)]
pub struct EnvNotReady;

pub trait Env<T> {
    fn reset(&mut self) -> T;
    fn step(&mut self, action: usize) -> Result<(T, f64, bool), EnvNotReady>;
    fn action_space(&self) -> ActionSpace;
}


pub use action_space::ActionSpace;
pub use blackjack::BlackJackEnv;
pub use frozen_lake::FrozenLakeEnv;