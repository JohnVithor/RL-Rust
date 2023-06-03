pub type ActiavationFunction = fn(&ndarray::Array2<f64>) -> ndarray::Array2<f64>;

pub fn tanh(x: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    return x.map(|v| v.tanh());
}

pub fn tanh_prime(x: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    return x.map(|v| 1.0-v.tanh().powf(2.0));
}