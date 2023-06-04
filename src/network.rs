/// Based on https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65
use self::layers::Layer;

pub mod activation;
pub mod layers;
pub mod loss;

pub struct Network {
    learning_rate: f64,
    layers: Vec<Box<dyn Layer>>,
    loss: Box<fn(&ndarray::Array2<f64>, &ndarray::Array2<f64>) -> Option<f64>>,
    loss_prime: Box<fn(&ndarray::Array2<f64>, &ndarray::Array2<f64>) -> ndarray::Array2<f64>>,
}

impl Network {
    pub fn new(
        learning_rate: f64,
        loss: Box<fn(&ndarray::Array2<f64>, &ndarray::Array2<f64>) -> Option<f64>>,
        loss_prime: Box<fn(&ndarray::Array2<f64>, &ndarray::Array2<f64>) -> ndarray::Array2<f64>>,
    ) -> Self {
        return Self {
            learning_rate,
            layers: vec![],
            loss,
            loss_prime,
        };
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
        return output;
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

        return match (self.loss)(&y_train, &output) {
            Some(v) => v,
            None => 0.0,
        };
    }
}
