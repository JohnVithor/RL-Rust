use std::fmt::Debug;
use std::hash::Hash;

use crate::network::Network;

use super::Policy;

pub type InputAdapter<T> = fn(T) -> ndarray::Array2<f64>;
pub type OutputAdapter<const COUNT: usize> = fn(ndarray::Array2<f64>) -> [f64; COUNT];
pub type InvOutputAdapter<const COUNT: usize> = fn([f64; COUNT]) -> ndarray::Array2<f64>;

#[derive(Debug)]
pub struct MainTargetNeuralPolicy<T, const COUNT: usize> {
    learning_rate: f64,
    input_adapter: InputAdapter<T>,
    target_network: Network,
    main_network: Network,
    output_adapter: OutputAdapter<COUNT>,
    inv_output_adapter: InvOutputAdapter<COUNT>,
    counter: u8,
}
impl<T: Hash + PartialEq + Eq + Clone + Debug, const COUNT: usize>
    MainTargetNeuralPolicy<T, COUNT>
{
    pub fn new(
        learning_rate: f64,
        input_adapter: InputAdapter<T>,
        network: Network,
        output_adapter: OutputAdapter<COUNT>,
        inv_output_adapter: InvOutputAdapter<COUNT>,
    ) -> Self {
        Self {
            learning_rate,
            input_adapter,
            target_network: network.clone(),
            main_network: network,
            output_adapter,
            inv_output_adapter,
            counter: 0,
        }
    }
}

impl<T: Hash + PartialEq + Eq + Clone + Debug, const COUNT: usize> Policy<T, COUNT>
    for MainTargetNeuralPolicy<T, COUNT>
{
    fn predict(&mut self, curr_obs: &T) -> [f64; COUNT] {
        let input = (self.input_adapter)(curr_obs.clone());
        let output = self.target_network.predict(input);
        (self.output_adapter)(output)
    }

    fn get_values(&mut self, curr_obs: &T) -> [f64; COUNT] {
        let input = (self.input_adapter)(curr_obs.clone());
        let output = self.main_network.predict(input);
        (self.output_adapter)(output)
    }

    fn update(&mut self, obs: &T, action: usize, next_obs: &T, temporal_difference: f64) -> f64 {
        let mut main_values: [f64; COUNT] = self.get_values(obs);
        let target_values: [f64; COUNT] = self.predict(next_obs);
        main_values[action] = target_values[action] + temporal_difference;
        let x = (self.input_adapter)(obs.clone());
        let y = (self.inv_output_adapter)(main_values);
        self.counter += 1;
        self.main_network.fit(x, y, self.learning_rate)
    }

    fn reset(&mut self) {
        self.main_network.reset();
        self.target_network.reset();
    }

    fn after_update(&mut self) {
        if self.counter >= 100 {
            self.target_network = self.main_network.clone();
            self.counter = 0;
        }
    }
}
