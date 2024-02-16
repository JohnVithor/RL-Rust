mod random_model;

pub use random_model::RandomExperienceReplay;

use std::fmt::Debug;

pub trait Model<T: Clone + Debug, A> {
    fn get_info(&self) -> (T, A, f32, T, A);
    fn add_info(&mut self, obs: T, action: A, reward: f32, next_obs: T, next_action: A);
    fn reset(&mut self);
}
