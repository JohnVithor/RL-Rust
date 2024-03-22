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

#[derive(Debug, Clone)]
pub enum Err {
    MethodNotFound,
    NotExpectedReturn,
    ExpectedItemNotFound,
    DifferentTypeExpected,
    ExpectedDataMissing,
}

impl PyEnv {
    pub fn new(env: Py<PyAny>) -> Self {
        Self { env }
    }

    fn extract_state(&self, resulting_tuple: &PyTuple) -> Result<Tensor, Err> {
        let Ok(start_obs) = resulting_tuple.get_item(0) else {
            return Err(Err::ExpectedItemNotFound);
        };
        let Ok(arr_data) = start_obs.extract::<PyReadonlyArrayDyn<f32>>() else {
            return Err(Err::DifferentTypeExpected);
        };
        let binding = arr_data.as_array();
        let Some(slice) = binding.as_slice() else {
            return Err(Err::ExpectedDataMissing);
        };
        Ok(Tensor::from_slice(slice))
    }

    fn extract_reward(&self, resulting_tuple: &PyTuple) -> Result<f32, Err> {
        let Ok(start_obs) = resulting_tuple.get_item(1) else {
            return Err(Err::ExpectedItemNotFound);
        };
        let Ok(reward) = start_obs.extract::<f32>() else {
            return Err(Err::DifferentTypeExpected);
        };
        Ok(reward)
    }

    fn extract_terminated(&self, resulting_tuple: &PyTuple) -> Result<bool, Err> {
        let Ok(start_obs) = resulting_tuple.get_item(2) else {
            return Err(Err::ExpectedItemNotFound);
        };
        let Ok(terminated) = start_obs.extract::<bool>() else {
            return Err(Err::DifferentTypeExpected);
        };
        Ok(terminated)
    }

    pub fn reset(&self, py: Python<'_>) -> Result<Tensor, Err> {
        let kwargs = [("seed", 0)].into_py_dict(py);
        let Ok(call_result) = self.env.call_method(py, "reset", (), Some(kwargs)) else {
            return Err(Err::MethodNotFound);
        };
        let Ok(resulting_tuple) = call_result.downcast::<PyTuple>(py) else {
            return Err(Err::NotExpectedReturn);
        };
        self.extract_state(resulting_tuple)
    }

    pub fn step(&self, py: Python<'_>, action: usize) -> Result<(Tensor, f32, bool), Err> {
        let Ok(call_result) = self
            .env
            .call_method(py, "step", PyTuple::new(py, [action]), None)
        else {
            return Err(Err::MethodNotFound);
        };
        let Ok(resulting_tuple) = call_result.downcast::<PyTuple>(py) else {
            return Err(Err::NotExpectedReturn);
        };
        let state = self.extract_state(resulting_tuple)?;
        let reward = self.extract_reward(resulting_tuple)?;
        let terminated = self.extract_terminated(resulting_tuple)?;
        Ok((state, reward, terminated))
    }
}

#[pyfunction]
pub fn test(env: Py<PyAny>) -> (Vec<f32>, f32, bool) {
    let a = PyEnv::new(env);
    let r = Python::with_gil(|py| a.reset(py));
    let r = Python::with_gil(|py| a.step(py, 0));
    let r = r.unwrap();
    let state = Vec::<f32>::try_from(r.0).expect("wrong type of tensor");
    let reward = r.1;
    let terminated = r.2;
    (state, reward, terminated)
}
