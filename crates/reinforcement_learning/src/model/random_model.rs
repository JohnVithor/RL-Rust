use rand::Rng;
use std::fmt::Debug;

use super::Model;

#[derive(Debug, Clone)]
pub struct RandomExperienceReplay<
    T: Default + Copy + Clone + Debug,
    A: Default + Copy + Clone,
    const SIZE: usize,
> {
    values: [(T, A, f32, T, A); SIZE],
    curr_size: usize,
    curr_pos: usize,
}

impl<T: Default + Copy + Clone + Debug, A: Default + Copy + Clone, const SIZE: usize> Default
    for RandomExperienceReplay<T, A, SIZE>
{
    fn default() -> Self {
        Self {
            values: [(T::default(), A::default(), 0.0, T::default(), A::default()); SIZE],
            curr_size: 0,
            curr_pos: 0,
        }
    }
}

impl<T: Default + Copy + Clone + Debug, A: Default + Copy + Clone, const SIZE: usize> Model<T, A>
    for RandomExperienceReplay<T, A, SIZE>
{
    fn get_info(&self) -> (T, A, f32, T, A) {
        let pos: usize = rand::thread_rng().gen_range(0..self.curr_size);
        self.values[pos]
    }

    fn add_info(&mut self, obs: T, action: A, reward: f32, next_obs: T, next_action: A) {
        self.values[self.curr_pos] = (obs, action, reward, next_obs, next_action);
        self.curr_pos = (self.curr_pos + 1) % SIZE;
        self.curr_size = (self.curr_size + 1).min(SIZE - 1);
    }

    fn reset(&mut self) {
        self.values = [(T::default(), A::default(), 0.0, T::default(), A::default()); SIZE];
        self.curr_pos = 0;
        self.curr_size = 0;
    }
}
