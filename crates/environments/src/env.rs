use ndarray::ArrayD;

use crate::space::SpaceInfo;

pub trait DiscreteActionEnv {
    type Error;
    fn reset(&mut self) -> Result<ArrayD<f32>, Self::Error>;
    fn step(&mut self, action: usize) -> Result<(ArrayD<f32>, f32, bool), Self::Error>;
    fn render(&self) -> String;
    fn observation_space(&self) -> SpaceInfo;
    fn action_space(&self) -> SpaceInfo;
}

pub trait ContinuousActionEnv {
    type Error;
    fn reset(&mut self) -> Result<ArrayD<f32>, Self::Error>;
    fn step(&mut self, action: ArrayD<f32>) -> Result<(ArrayD<f32>, f32, bool), Self::Error>;
    fn render(&self) -> String;
    fn observation_space(&self) -> SpaceInfo;
    fn action_space(&self) -> SpaceInfo;
}
