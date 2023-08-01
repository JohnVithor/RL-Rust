mod random_model;

pub use random_model::RandomExperienceReplay;

use std::fmt::Debug;

pub trait Model<T: Clone + Debug, const COUNT: usize> {
    fn get_info(&self) -> (T, usize, f64, T, usize);
    fn add_info(&mut self, obs: T, action: usize, reward: f64, next_obs: T, next_action: usize);
    fn reset(&mut self);
}
