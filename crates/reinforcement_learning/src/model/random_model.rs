use rand::Rng;
use std::fmt::Debug;

use super::Model;

#[derive(Debug, Clone)]
pub struct RandomExperienceReplay<
    T: Default + Copy + Clone + Debug,
    const COUNT: usize,
    const SIZE: usize,
> {
    values: [(T, usize, f64, T, usize); SIZE],
    curr_size: usize,
    curr_pos: usize,
}

impl<T: Default + Copy + Clone + Debug, const COUNT: usize, const SIZE: usize> Default
    for RandomExperienceReplay<T, COUNT, SIZE>
{
    fn default() -> Self {
        Self {
            values: [(T::default(), 0, 0.0, T::default(), 0); SIZE],
            curr_size: 0,
            curr_pos: 0,
        }
    }
}

impl<T: Default + Copy + Clone + Debug, const COUNT: usize, const SIZE: usize> Model<T, COUNT>
    for RandomExperienceReplay<T, COUNT, SIZE>
{
    fn get_info(&self) -> (T, usize, f64, T, usize) {
        let pos: usize = rand::thread_rng().gen_range(0..self.curr_size);
        self.values[pos]
    }

    fn add_info(&mut self, obs: T, action: usize, reward: f64, next_obs: T, next_action: usize) {
        self.values[self.curr_pos] = (obs, action, reward, next_obs, next_action);
        self.curr_pos = (self.curr_pos + 1) % SIZE;
        self.curr_size = (self.curr_size + 1).min(SIZE - 1);
    }

    fn reset(&mut self) {
        self.values = [(T::default(), 0, 0.0, T::default(), 0); SIZE];
        self.curr_pos = 0;
        self.curr_size = 0;
    }
}
