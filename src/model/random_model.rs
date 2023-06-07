use fxhash::FxBuildHasher;
use indexmap::IndexMap;
use rand::Rng;
use std::fmt::Debug;
use std::hash::Hash;

use super::Model;

#[derive(Debug, Clone)]
pub struct RandomModel<T: Hash + PartialEq + Eq + Clone + Debug, const COUNT: usize> {
    values: IndexMap<(T, usize), (T, f64), FxBuildHasher>,
}

impl<T: Hash + PartialEq + Eq + Clone + Debug, const COUNT: usize> Default
    for RandomModel<T, COUNT>
{
    fn default() -> Self {
        Self {
            values: IndexMap::default(),
        }
    }
}

impl<T: Hash + PartialEq + Eq + Clone + Debug, const COUNT: usize> Model<T, COUNT>
    for RandomModel<T, COUNT>
{
    fn get_info(&self) -> (T, usize, T, f64) {
        let rand = self
            .values
            .get_index(rand::thread_rng().gen_range(0..self.values.len()));
        let (state, action) = rand.unwrap().0.clone();
        let (next_state, reward) = rand.unwrap().1.clone();
        (state, action, next_state, reward)
    }

    fn add_info(&mut self, obs: T, action: usize, reward: f64, next_obs: T) {
        self.values
            .entry((obs, action))
            .or_insert((next_obs, reward));
    }

    fn reset(&mut self) {
        self.values = IndexMap::default();
    }
}
