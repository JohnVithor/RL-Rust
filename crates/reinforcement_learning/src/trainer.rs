mod continuous_discrete_trainer;
// mod discrete_discrete_trainer;
pub use continuous_discrete_trainer::CDTrainer;
// pub use discrete_discrete_trainer::DDTrainer;

pub type TrainResults = (Vec<f32>, Vec<u128>, Vec<f32>, Vec<f32>, Vec<f32>);
