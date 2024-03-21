use environments::EnvError;
use numpy::PyReadonlyArrayDyn;
use pyo3::{
    pyfunction,
    types::{IntoPyDict, PyTuple},
    Py, PyAny, Python,
};
use tch::Tensor;

pub struct PyEnv {
    env: Py<PyAny>,
}

impl PyEnv {
    pub fn new(env: Py<PyAny>) -> Self {
        Self { env }
    }

    pub fn reset(&self, py: Python<'_>) -> Result<Tensor, EnvError> {
        let kwargs = [("seed", 0)].into_py_dict(py);
        if let Ok(call_result) = self.env.call_method(py, "reset", (), Some(kwargs)) {
            if let Ok(resulting_tuple) = call_result.downcast::<PyTuple>(py) {
                if let Ok(start_obs) = resulting_tuple.get_item(0) {
                    if let Ok(arr_data) = start_obs.extract::<PyReadonlyArrayDyn<f32>>() {
                        if let Some(slice) = arr_data.as_array().as_slice() {
                            println!("{slice:?}");
                            return Ok(Tensor::from_slice(slice));
                        }
                    }
                }
            }
        }
        Err(EnvError::EnvNotReady)
    }

    pub fn step(&self, action: usize, py: Python<'_>) -> Result<(Tensor, f32, bool), EnvError> {
        Err(EnvError::EnvNotReady)
    }
}

#[pyfunction]
pub fn test(env: Py<PyAny>) -> Vec<f32> {
    let a = PyEnv::new(env);
    let r = Python::with_gil(|py| a.reset(py));
    Vec::<f32>::try_from(r.unwrap()).expect("wrong type of tensor")
}
