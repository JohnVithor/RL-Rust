use fxhash::FxBuildHasher;
use indexmap::IndexMap;
use rand::Rng;
use std::fmt::Debug;
use std::hash::Hash;

use super::Model;

#[derive(Debug, Clone)]
pub struct RandomModel<T: Hash + PartialEq + Eq + Clone + Debug, const COUNT: usize> {
    values: IndexMap<(T, usize), (T, f64, usize), FxBuildHasher>,
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
    fn get_info(&self) -> (T, usize, f64, T, usize) {
        let rand = self
            .values
            .get_index(rand::thread_rng().gen_range(0..self.values.len()));
        let (state, action) = rand.unwrap().0.clone();
        let (next_state, reward, next_action) = rand.unwrap().1.clone();
        (state, action, reward, next_state, next_action)
    }

    fn add_info(&mut self, obs: T, action: usize, reward: f64, next_obs: T, next_action: usize) {
        self.values
            .entry((obs, action))
            .or_insert((next_obs, reward, next_action));
    }

    fn reset(&mut self) {
        self.values = IndexMap::default();
    }
}
