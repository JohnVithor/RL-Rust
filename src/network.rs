use std::fmt::Debug;

/// Based on https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65
use self::layers::Layer;

pub mod activation;
pub mod layers;
pub mod loss;

pub struct Network {
    learning_rate: f64,
    layers: Vec<Box<dyn Layer>>,
    loss: fn(&ndarray::Array2<f64>, &ndarray::Array2<f64>) -> Option<f64>,
    loss_prime: fn(&ndarray::Array2<f64>, &ndarray::Array2<f64>) -> ndarray::Array2<f64>,
}

impl Clone for Network {
    fn clone(&self) -> Self {
        Self {
            learning_rate: self.learning_rate.clone(),
            layers: self.layers.clone(),
            loss: self.loss.clone(),
            loss_prime: self.loss_prime.clone(),
        }
    }
}

impl Debug for Network {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Network")
            .field("learning_rate", &self.learning_rate)
            .field("layers", &self.layers)
            .finish()
    }
}

impl Network {
    pub fn new(
        learning_rate: f64,
        loss: fn(&ndarray::Array2<f64>, &ndarray::Array2<f64>) -> Option<f64>,
        loss_prime: fn(&ndarray::Array2<f64>, &ndarray::Array2<f64>) -> ndarray::Array2<f64>,
    ) -> Self {
        Self {
            learning_rate,
            layers: vec![],
            loss,
            loss_prime,
        }
    }

    // add layer to network
    pub fn add(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer)
    }

    // predict output for given input
    pub fn predict(&mut self, input: ndarray::Array2<f64>) -> ndarray::Array2<f64> {
        // forward propagation
        let mut output = input;
        for layer in &mut self.layers {
            output = layer.forward_propagation(output);
        }
        output
    }

    // train the network
    pub fn fit(&mut self, x_train: ndarray::Array2<f64>, y_train: ndarray::Array2<f64>) -> f64 {
        // forward propagation
        let mut output = x_train;
        for layer in &mut self.layers {
            output = layer.forward_propagation(output);
        }

        // backward propagation
        let mut error = (self.loss_prime)(&y_train, &output);
        for layer in self.layers.iter_mut().rev() {
            error = layer.backward_propagation(error, self.learning_rate)
        }

        (self.loss)(&y_train, &output).unwrap_or(0.0)
    }

    pub fn copy_weights_and_bias(&mut self, other: &Network) {
        assert!(self.layers.len() == other.layers.len());
        for (my_layer, other_layer) in self.layers.iter_mut().zip(&other.layers) {
            my_layer.copy_weights_and_bias(other_layer.as_ref())
        }
    }

    pub fn reset(&mut self) {
        for l in &mut self.layers {
            l.reset()
        }
    }
}
