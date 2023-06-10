use std::fmt::Debug;

use ndarray::arr2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

pub trait Layer: LayerClone {
    // computes the output Y of a layer for a given input X
    fn forward_propagation(&mut self, input: ndarray::Array2<f64>) -> ndarray::Array2<f64>;

    // computes dE/dX for a given dE/dY (and update parameters if any)
    fn backward_propagation(
        &mut self,
        output_error: ndarray::Array2<f64>,
        learning_rate: f64,
    ) -> ndarray::Array2<f64>;

    fn get_weights(&self) -> ndarray::Array2<f64>;
    fn get_bias(&self) -> ndarray::Array2<f64>;

    fn set_weights(&mut self, weights: ndarray::Array2<f64>);
    fn set_bias(&mut self, bias: ndarray::Array2<f64>);

    fn copy_weights_and_bias(&mut self, other: &dyn Layer) {
        self.set_weights(other.get_weights());
        self.set_bias(other.get_bias());
    }

    fn reset(&mut self);
}

pub trait LayerClone {
    fn clone_box(&self) -> Box<dyn Layer>;
}

impl<T> LayerClone for T
where
    T: 'static + Layer + Clone,
{
    fn clone_box(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn Layer> {
    fn clone(&self) -> Box<dyn Layer> {
        self.clone_box()
    }
}

impl Debug for dyn Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Layer").finish()
    }
}

#[derive(Clone)]
pub struct DenseLayer {
    input: ndarray::Array2<f64>,
    weights: ndarray::Array2<f64>,
    bias: ndarray::Array2<f64>,
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let input = ndarray::Array2::zeros((input_size, output_size));
        // let weights = ndarray::Array::random((input_size, output_size), Uniform::new(0., 1.));
        // let bias = ndarray::Array::random((1, output_size), Uniform::new(0., 1.));
        let weights = ndarray::Array2::ones((input_size, output_size)) * 0.5;
        let bias = ndarray::Array2::ones((1, output_size));
        println!("w {:?}", weights);
        println!("b {:?}", bias);
        Self {
            input,
            weights,
            bias,
        }
    }
}

impl Layer for DenseLayer {
    fn forward_propagation(&mut self, input: ndarray::Array2<f64>) -> ndarray::Array2<f64> {
        self.input = input;
        self.input.dot(&self.weights) + &self.bias
    }

    fn backward_propagation(
        &mut self,
        output_error: ndarray::Array2<f64>,
        learning_rate: f64,
    ) -> ndarray::Array2<f64> {
        let input_error = output_error.dot(&self.weights.t());
        let weights_error = self.input.t().dot(&output_error);
        self.weights = &self.weights - learning_rate * weights_error;
        self.bias = &self.bias - learning_rate * output_error;
        input_error
    }
    fn reset(&mut self) {
        self.input = ndarray::Array2::zeros(self.input.raw_dim());
        self.weights = ndarray::Array::random(self.weights.raw_dim(), Uniform::new(0., 1.));
        self.bias = ndarray::Array::random(self.bias.raw_dim(), Uniform::new(0., 1.));
    }

    fn get_weights(&self) -> ndarray::Array2<f64> {
        self.weights.clone()
    }

    fn get_bias(&self) -> ndarray::Array2<f64> {
        self.bias.clone()
    }

    fn set_weights(&mut self, weights: ndarray::Array2<f64>) {
        self.weights = weights
    }

    fn set_bias(&mut self, bias: ndarray::Array2<f64>) {
        self.bias = bias;
    }
}

#[derive(Clone)]
pub struct ActivationLayer {
    input: ndarray::Array2<f64>,
    activation: fn(&ndarray::Array2<f64>) -> ndarray::Array2<f64>,
    activation_prime: fn(&ndarray::Array2<f64>) -> ndarray::Array2<f64>,
}

impl ActivationLayer {
    pub fn new(
        activation: fn(&ndarray::Array2<f64>) -> ndarray::Array2<f64>,
        activation_prime: fn(&ndarray::Array2<f64>) -> ndarray::Array2<f64>,
    ) -> Self {
        let input = ndarray::Array2::zeros((0, 0));
        Self {
            input,
            activation,
            activation_prime,
        }
    }
}

impl Layer for ActivationLayer {
    fn forward_propagation(&mut self, input: ndarray::Array2<f64>) -> ndarray::Array2<f64> {
        self.input = input;
        (self.activation)(&self.input)
    }

    fn backward_propagation(
        &mut self,
        output_error: ndarray::Array2<f64>,
        _learning_rate: f64,
    ) -> ndarray::Array2<f64> {
        (self.activation_prime)(&self.input) * output_error
    }

    fn reset(&mut self) {}

    fn get_weights(&self) -> ndarray::Array2<f64> {
        arr2(&[[0.0]])
    }

    fn get_bias(&self) -> ndarray::Array2<f64> {
        arr2(&[[0.0]])
    }

    fn set_weights(&mut self, _weights: ndarray::Array2<f64>) {}

    fn set_bias(&mut self, _bias: ndarray::Array2<f64>) {}
}
