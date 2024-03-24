use pyo3::{
    create_exception, exceptions::PyException, pymodule, types::PyModule, wrap_pyfunction,
    PyResult, Python,
};
use test::PyCounter;

mod agent;
mod env;
mod test;

/// A Python module implemented in Rust.
#[pymodule]
fn rust_drl(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", "0.0.1")?;
    m.add_class::<agent::DQNAgent>()?;
    m.add_class::<PyCounter>()?;
    m.add_function(wrap_pyfunction!(env::test, m)?).unwrap();
    create_exception!(m, PyEnvErr, PyException);
    Ok(())
}
