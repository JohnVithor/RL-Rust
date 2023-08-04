use std::fmt::Debug;

#[derive(Debug, Clone)]
pub struct EnvNotReady;

pub trait Action {
    const SIZE: usize;
}

pub trait ContinuousAction: Action + From<f64> {
    const RANGE: (f64, f64);
}

pub trait DiscreteAction: Action + From<usize> {
    const RANGE: usize;
}

impl Action for usize {
    const SIZE: usize = 1;
}

impl Action for f64 {
    const SIZE: usize = 1;
}

pub trait Env<T: Debug, A: Action + Debug> {
    fn reset(&mut self) -> T;
    fn step(&mut self, action: A) -> Result<(T, f64, bool), EnvNotReady>;
    fn render(&self) -> String;
}
