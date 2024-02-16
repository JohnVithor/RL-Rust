use std::fmt::Debug;

use crate::space::SpaceInfo;

#[derive(Debug, Clone)]
pub enum EnvError {
    EnvNotReady,
}

pub trait Env<T, A> {
    fn reset(&mut self) -> T;
    fn step(&mut self, action: A) -> Result<(T, f32, bool), EnvError>;
    fn render(&self) -> String;
    fn observation_space(&self) -> SpaceInfo;
    fn action_space(&self) -> SpaceInfo;
}
