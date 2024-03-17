use pyo3::{pymodule, types::PyModule, PyResult, Python};

mod agent;

/// A Python module implemented in Rust.
#[pymodule]
fn rust_drl(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", "0.0.1")?;
    m.add_class::<agent::DQNAgent>()?;
    Ok(())
}
