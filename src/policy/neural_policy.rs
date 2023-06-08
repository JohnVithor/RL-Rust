use std::fmt::Debug;
use std::hash::Hash;

use crate::network::Network;

use super::Policy;

pub type InputAdapter<T> = fn(T) -> ndarray::Array2<f64>;
pub type OutputAdapter<const COUNT: usize> = fn(ndarray::Array2<f64>) -> [f64; COUNT];
pub type InvOutputAdapter<const COUNT: usize> = fn([f64; COUNT]) -> ndarray::Array2<f64>;

pub struct NeuralPolicy<T, const COUNT: usize> {
    input_adapter: InputAdapter<T>,
    network: Network,
    output_adapter: OutputAdapter<COUNT>,
    inv_output_adapter: InvOutputAdapter<COUNT>,
}
impl<T: Hash + PartialEq + Eq + Clone + Debug, const COUNT: usize> NeuralPolicy<T, COUNT> {
    pub fn new(
        input_adapter: InputAdapter<T>,
        network: Network,
        output_adapter: OutputAdapter<COUNT>,
        inv_output_adapter: InvOutputAdapter<COUNT>,
    ) -> Self {
        Self {
            input_adapter,
            network,
            output_adapter,
            inv_output_adapter,
        }
    }
}

impl<T: Hash + PartialEq + Eq + Clone + Debug, const COUNT: usize> Policy<T, COUNT>
    for NeuralPolicy<T, COUNT>
{
    fn predict(&mut self, curr_obs: &T) -> [f64; COUNT] {
        let input = (self.input_adapter)(curr_obs.clone());
        let output = self.network.predict(input);
        (self.output_adapter)(output)
    }

    fn get_values(&mut self, curr_obs: &T) -> [f64; COUNT] {
        self.predict(curr_obs)
    }

    fn update(&mut self, obs: &T, action: usize, temporal_difference: f64) {
        let mut values = self.predict(obs);
        values[action] += temporal_difference;
        let y = (self.inv_output_adapter)(values);
        let x = (self.input_adapter)(obs.clone());
        self.network.fit(x, y);
    }

    fn reset(&mut self) {
        self.network.reset();
    }

    fn after_update(&mut self) {}
}
