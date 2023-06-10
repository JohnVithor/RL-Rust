use crate::utils::ndarray_max;

pub fn tanh(x: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    x.map(|v| v.tanh())
}

pub fn tanh_prime(x: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    x.map(|v| 1.0 - v.tanh().powf(2.0))
}

pub fn softmax(x: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    let max = ndarray_max(x);
    let e = x.map(|v| (v-max).exp());
    e.clone() / e.sum()
}
