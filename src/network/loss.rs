pub fn mse(y_true: &ndarray::Array2<f64>, y_pred: &ndarray::Array2<f64>) -> Option<f64> {
    return (y_true-y_pred).map(|v| v.powf(2.0)).mean();
}
pub fn mse_prime(y_true: &ndarray::Array2<f64>, y_pred: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    return 2.0 * (y_pred-y_true) / (y_true.len() as f64);
}