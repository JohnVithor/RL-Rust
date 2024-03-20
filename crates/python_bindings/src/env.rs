use environments::EnvError;
use numpy::{
    datetime::{units, Timedelta},
    Complex64, IntoPyArray, PyArray1, PyArrayDyn, PyReadonlyArray1, PyReadonlyArrayDyn,
    PyReadwriteArray1, PyReadwriteArrayDyn,
};
use pyo3::types::PyDict;
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    types::PyTuple,
};
use pyo3::{prelude::*, types::IntoPyDict};
use pyo3::{pyfunction, Py, PyAny, Python};
pub use tch;
use tch::Tensor;
pub struct PyTensor(pub tch::Tensor);

impl std::ops::Deref for PyTensor {
    type Target = tch::Tensor;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub fn wrap_tch_err(err: tch::TchError) -> PyErr {
    PyErr::new::<PyValueError, _>(format!("{err:?}"))
}

impl<'source> FromPyObject<'source> for PyTensor {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let ptr = ob.as_ptr() as *mut tch::python::CPyObject;
        let tensor = unsafe { tch::Tensor::pyobject_unpack(ptr) };
        tensor
            .map_err(wrap_tch_err)?
            .ok_or_else(|| {
                let type_ = ob.get_type();
                PyErr::new::<PyTypeError, _>(format!("expected a torch.Tensor, got {type_}"))
            })
            .map(PyTensor)
    }
}

impl IntoPy<PyObject> for PyTensor {
    fn into_py(self, py: Python<'_>) -> PyObject {
        // There is no fallible alternative to ToPyObject/IntoPy at the moment so we return
        // None on errors. https://github.com/PyO3/pyo3/issues/1813
        self.0.pyobject_wrap().map_or_else(
            |_| py.None(),
            |ptr| unsafe { PyObject::from_owned_ptr(py, ptr as *mut pyo3::ffi::PyObject) },
        )
    }
}

pub struct PyEnv {
    env: Py<PyAny>,
}

impl PyEnv {
    pub fn new(env: Py<PyAny>) -> Self {
        Self { env }
    }

    pub fn reset(&self, py: Python<'_>) -> Result<(Tensor, f32, bool), EnvError> {
        let kwargs = [("seed", 0)].into_py_dict(py);
        let r = self.env.call_method(py, "reset", (), Some(kwargs));
        match r {
            Ok(data) => {
                if let Ok(tuple) = data.downcast::<PyTuple>(py) {
                    let b = tuple.get_item(0).unwrap();
                    let obs = b.extract::<PyReadonlyArrayDyn<f32>>().unwrap();
                    let q = obs.as_array();
                    println!("{q:?}");
                }
            }
            Err(e) => {
                println!("{e}")
            }
        }
        Err(EnvError::EnvNotReady)
    }

    pub fn step(&self, action: Tensor, py: Python<'_>) -> Result<(Tensor, f32, bool), EnvError> {
        Err(EnvError::EnvNotReady)
    }
}

#[pyfunction]
pub fn test(env: Py<PyAny>) -> String {
    let a = PyEnv::new(env);
    let _ = Python::with_gil(|py| a.reset(py));
    "Foi".to_string()
}
