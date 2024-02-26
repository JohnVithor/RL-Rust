mod continuous_observation_trainer;
mod full_discrete_trainer;
pub use continuous_observation_trainer::ContinuousObsDiscreteTrainer;
pub use full_discrete_trainer::FullDiscreteTrainer;

pub type TrainResults = (Vec<f32>, Vec<u128>, Vec<f32>, Vec<f32>, Vec<f32>);
