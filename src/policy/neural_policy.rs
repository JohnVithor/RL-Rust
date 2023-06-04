use std::hash::Hash;

use crate::{env::ActionSpace, network::Network};

use super::Policy;

pub type InputAdapter<T> = fn (T) -> ndarray::Array2<f64>;
pub type OutputAdapter = fn (ndarray::Array2<f64>) -> ndarray::Array1<f64>;
pub type InvOutputAdapter = fn (ndarray::Array1<f64>) -> ndarray::Array2<f64>;

pub struct NeuralPolicy<'a, T> {
    input_adapter: InputAdapter<T>,
    network: Network<'a>,
    output_adapter: OutputAdapter,
    inv_output_adapter: InvOutputAdapter,
    discount_factor: f64,
    action_space: ActionSpace,
}
impl<'a, T: Hash+PartialEq+Eq+Clone> NeuralPolicy<'a, T> {
    pub fn new(
        input_adapter: InputAdapter<T>, 
        network: Network<'a>, 
        output_adapter: OutputAdapter,
        inv_output_adapter: InvOutputAdapter,
        discount_factor: f64,
        action_space: ActionSpace,
    ) -> Self {
        return Self {
            input_adapter,
            network,
            output_adapter,
            inv_output_adapter,
            discount_factor,
            action_space
        };
    }
}

impl<'a, T: Hash+PartialEq+Eq+Clone> Policy<T> for NeuralPolicy<'a, T>{
    fn get_values(&mut self, curr_obs: T) -> ndarray::Array1<f64> {
        return self.predict(curr_obs);
    }

    fn predict(&mut self, curr_obs: T) -> ndarray::Array1<f64> {
        let input = (self.input_adapter)(curr_obs);
        let output = self.network.predict(input);
        return (self.output_adapter)(output);
    }

    fn update_values(&mut self, curr_obs: T, curr_action: usize, _next_obs: T, _next_action: usize, temporal_difference: f64) {
        let mut values = self.predict(curr_obs.clone());
        values[curr_action] = values[curr_action] + temporal_difference;
        let y = (self.inv_output_adapter)(values);
        let x = (self.input_adapter)(curr_obs);
        self.network.fit(x, y);
    }

    fn get_action_space(&self) -> &ActionSpace {
        return &self.action_space;
    }

    fn reset(&mut self) {
        self.network.reset();
    }

    fn after_update(&mut self) {
        
    }

    fn get_learning_rate(&self) -> f64 {
        return self.network.get_learning_rate();
    }

    fn get_discount_factor(&self) -> f64 {
        return self.discount_factor;
    }

    fn get_td(&mut self, curr_obs: T, curr_action: usize, reward: f64, future_q_value: f64) -> f64 {
        let values = self.get_values(curr_obs);
        let temporal_difference: f64 = reward + self.discount_factor * future_q_value - values[curr_action];
        return temporal_difference;
    }
}
