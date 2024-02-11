use std::fmt::Debug;

#[derive(Debug, Clone)]
pub enum EnvError {
    EnvNotReady,
}

pub trait DiscreteAction: From<usize> + Clone + Into<usize>
where
    Self: Sized,
{
    const RANGE: usize = { std::mem::variant_count::<Self>() };
}

pub trait ContinuousEnv<T, const N: usize> {
    fn reset(&mut self) -> T;
    fn step(&mut self, actions: [f64; N]) -> Result<(T, f64, bool), EnvError>;
    fn render(&self) -> String;
}

pub trait DiscreteEnv<T, A: DiscreteAction> {
    fn reset(&mut self) -> T;
    fn step(&mut self, action: A) -> Result<(T, f64, bool), EnvError>;
    fn render(&self) -> String;
}

pub trait MixedEnv<T, A: DiscreteAction, const N: usize> {
    fn reset(&mut self) -> T;
    fn step(&mut self, d_action: A, c_actions: [f64; N]) -> Result<(T, f64, bool), EnvError>;
    fn render(&self) -> String;
}
