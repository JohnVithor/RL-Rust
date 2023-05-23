mod action_space;
mod observation;
mod blackjack;

#[derive(Debug, Clone)]
pub struct EnvNotReady;

pub trait Env {
    fn reset(&mut self) -> Observation;
    fn step(&mut self, action: usize) -> Result<(Observation, f64, bool), EnvNotReady>;
    fn action_space(&self) -> ActionSpace;
}


pub use action_space::ActionSpace;
pub use observation::Observation;
pub use blackjack::BlackJackEnv;