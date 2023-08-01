use crate::utils::ndarray_max;

// https://www.v7labs.com/blog/neural-networks-activation-functions

pub fn linear(x: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    x.clone()
}

pub fn linear_prime(x: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    ndarray::Array2::ones(x.raw_dim())
}

pub fn tanh(x: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    x.map(|v| v.tanh())
}

pub fn tanh_prime(x: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    x.map(|v| 1.0 - v.tanh().powf(2.0))
}

pub fn relu(x: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    x.map(|v| v.max(0.0))
}

pub fn relu_prime(x: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    x.map(|v| if v > &0.0 { 1.0 } else { 0.0 })
}

pub fn leaky_relu(x: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    x.map(|v| v.max(0.1 * v))
}

pub fn leaky_relu_prime(x: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    x.map(|v| if v > &0.0 { 1.0 } else { 0.01 })
}

pub fn relu6(x: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    x.map(|v| v.max(0.0).min(6.0))
}

pub fn relu6_prime(x: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    x.map(|v| if v > &0.0 && v < &6.0 { 1.0 } else { 0.0 })
}

pub fn leaky_relu6(x: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    x.map(|v| v.max(0.1 * v).min(6.0))
}

pub fn leaky_relu6_prime(x: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    x.map(|v| if v > &0.0 && v < &6.0 { 1.0 } else { 0.01 })
}

pub fn sigmoid(x: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    x.map(|v| 1.0 / (1.0 + (-v).exp()))
}

pub fn sigmoid_prime(x: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    let s = sigmoid(x);
    &s * (1.0 - &s)
}

pub fn softmax(x: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    let max = ndarray_max(x);
    let e = x.map(|v| (v - max).exp());
    e.clone() / e.sum()
}

pub fn softmax_prime(x: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    let max = ndarray_max(x);
    let e = x.map(|v| (v - max).exp());
    e.clone() / e.sum()
}

pub fn swish(x: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    x * sigmoid(x)
}

pub fn swish_prime(x: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    x.map(|v| (v.exp() * (v + v.exp() + 1.0)) / ((v.exp() + 1.0) * (v.exp() + 1.0)))
}

pub fn hard_swish(x: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    x * relu6(&(x + 3.0)) / 6.0
}

pub fn hard_swish_prime(x: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    x.map(|v| {
        if v > &-3.0 {
            (2.0 * v + 3.0) / 6.0
        } else {
            0.0
        }
    })
}
