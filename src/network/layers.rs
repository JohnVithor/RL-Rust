use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

pub trait Layer {
    // computes the output Y of a layer for a given input X
    fn forward_propagation(&mut self, input: ndarray::Array2<f64> ) -> ndarray::Array2<f64>;
    // computes dE/dX for a given dE/dY (and update parameters if any)
    fn backward_propagation(&mut self, output_error: ndarray::Array2<f64>, learning_rate: f64) -> ndarray::Array2<f64>;
}

pub struct DenseLayer {
    input: ndarray::Array2<f64>,
    weights: ndarray::Array2<f64>,
    bias: ndarray::Array2<f64>
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let input = ndarray::Array2::zeros((input_size, output_size));
        let weights = ndarray::Array::random((input_size, output_size), Uniform::new(0., 1.));
        let bias = ndarray::Array::random((1, output_size), Uniform::new(0., 1.));
        return Self { input, weights, bias }
    }
}

impl Layer for DenseLayer {
    fn forward_propagation(&mut self, input: ndarray::Array2<f64> ) -> ndarray::Array2<f64> {
        self.input = input;
        return self.input.dot(&self.weights) + &self.bias
    }

    fn backward_propagation(&mut self, output_error: ndarray::Array2<f64>, learning_rate: f64) -> ndarray::Array2<f64>{
        let input_error = output_error.dot(&self.weights.t());
        let weights_error = self.input.t().dot(&output_error);
        self.weights = &self.weights - learning_rate * weights_error;
        self.bias = &self.bias - learning_rate * output_error;
        return input_error;
    }
}

pub struct ActivationLayer {
    input: ndarray::Array2<f64>,
    activation: Box<dyn Fn(&ndarray::Array2<f64>) -> ndarray::Array2<f64>>,
    activation_prime: Box<dyn Fn(&ndarray::Array2<f64>) -> ndarray::Array2<f64>>
} 

impl ActivationLayer {
    pub fn new(
        activation: Box<dyn Fn(&ndarray::Array2<f64>) -> ndarray::Array2<f64>>,
        activation_prime: Box<dyn Fn(&ndarray::Array2<f64>) -> ndarray::Array2<f64>>
    ) -> Self {
        let input = ndarray::Array2::zeros((0, 0));
        return Self{ input, activation, activation_prime};
    }
}

impl Layer for ActivationLayer {
    fn forward_propagation(&mut self, input: ndarray::Array2<f64> ) -> ndarray::Array2<f64> {
        self.input = input;
        return (self.activation)(&self.input);
    }

    fn backward_propagation(&mut self, output_error: ndarray::Array2<f64>, _learning_rate: f64) -> ndarray::Array2<f64> {
        return (self.activation_prime)(&self.input) * output_error
    }
}