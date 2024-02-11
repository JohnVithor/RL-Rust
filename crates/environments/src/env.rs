use std::fmt::Debug;

#[derive(Debug, Clone)]
pub enum EnvError {
    EnvNotReady,
}

pub trait ContinuousEnv<T, const N: usize> {
    fn reset(&mut self) -> T;
    fn step(&mut self, actions: [f64; N]) -> Result<(T, f64, bool), EnvError>;
    fn render(&self) -> String;
}

pub trait DiscreteEnv<T, A> {
    fn reset(&mut self) -> T;
    fn step(&mut self, action: A) -> Result<(T, f64, bool), EnvError>;
    fn render(&self) -> String;
}

pub trait MixedEnv<T, A, const N: usize> {
    fn reset(&mut self) -> T;
    fn step(&mut self, d_action: A, c_actions: [f64; N]) -> Result<(T, f64, bool), EnvError>;
    fn render(&self) -> String;
}
