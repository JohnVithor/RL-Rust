use environments::{
    space::{SpaceInfo, SpaceTypeBounds},
    DiscreteActionEnv,
};
use numpy::{
    ndarray::{Array1, ArrayD},
    PyReadonlyArrayDyn,
};
use pyo3::{exceptions::PyTypeError, pyfunction, types::PyTuple, Py, PyAny, PyResult, Python};
use rand::{rngs::StdRng, RngCore, SeedableRng};
use reinforcement_learning::{
    action_selection::epsilon_greedy::{EpsilonGreedy, EpsilonUpdateStrategy},
    agent::{DoubleDeepAgent, OptimizerEnum},
    experience_buffer::RandomExperienceBuffer,
};
use reinforcement_learning::{agent::TrainResults, trainer::CDTrainer};
use std::time::Instant;
use tch::{
    nn::{self, Module, VarStore},
    Device, Kind,
};
pub struct PyEnv {
    env: Py<PyAny>,
}

#[derive(Debug, Clone)]
pub enum PyEnvErr {
    MethodNotFound,
    NotExpectedReturn,
    ExpectedItemNotFound,
    DifferentTypeExpected,
    ExpectedDataMissing,
}

impl PyEnv {
    pub fn new(env: Py<PyAny>) -> PyResult<Self> {
        Python::with_gil(|py: Python<'_>| -> PyResult<Self> {
            let a = env.as_ref(py);
            if !a.hasattr("reset").unwrap() {
                return Err(PyTypeError::new_err(
                    "Object hasn't 'reset' method!".to_string(),
                ));
            }
            if !a.hasattr("step").unwrap() {
                return Err(PyTypeError::new_err(
                    "Object hasn't 'step' method!".to_string(),
                ));
            }
            if !a.hasattr("action_space").unwrap() {
                return Err(PyTypeError::new_err(
                    "Object hasn't 'action_space' attribute!".to_string(),
                ));
            }
            Ok(Self { env })
        })
    }

    fn extract_state(&self, resulting_tuple: &PyTuple) -> Result<ArrayD<f32>, PyEnvErr> {
        let Ok(start_obs) = resulting_tuple.get_item(0) else {
            return Err(PyEnvErr::ExpectedItemNotFound);
        };
        match start_obs.extract::<PyReadonlyArrayDyn<f32>>() {
            Ok(arr_data) => {
                let binding = arr_data.as_array();
                let Some(slice) = binding.as_slice() else {
                    return Err(PyEnvErr::ExpectedDataMissing);
                };
                Ok(Array1::from_vec(Vec::from(slice)).into_dyn())
            }
            Err(_) => {
                let Ok(elem) = start_obs.extract::<usize>() else {
                    return Err(PyEnvErr::DifferentTypeExpected);
                };
                Ok(Array1::from_elem(1, elem as f32).into_dyn())
            }
        }
    }

    fn extract_reward(&self, resulting_tuple: &PyTuple) -> Result<f32, PyEnvErr> {
        let Ok(start_obs) = resulting_tuple.get_item(1) else {
            return Err(PyEnvErr::ExpectedItemNotFound);
        };
        let Ok(reward) = start_obs.extract::<f32>() else {
            return Err(PyEnvErr::DifferentTypeExpected);
        };
        Ok(reward)
    }

    fn extract_terminated(&self, resulting_tuple: &PyTuple) -> Result<bool, PyEnvErr> {
        let Ok(start_obs) = resulting_tuple.get_item(2) else {
            return Err(PyEnvErr::ExpectedItemNotFound);
        };
        let Ok(terminated) = start_obs.extract::<bool>() else {
            return Err(PyEnvErr::DifferentTypeExpected);
        };
        Ok(terminated)
    }

    fn extract_space(&self, attribute: Py<PyAny>, py: Python<'_>) -> SpaceInfo {
        let space = attribute.as_ref(py).get_type().getattr("__name__").unwrap();
        let name: String = space.extract().unwrap();
        if name.eq("Discrete") {
            let size = attribute.as_ref(py).getattr("n").unwrap();
            SpaceInfo::new(vec![SpaceTypeBounds::Discrete(size.extract().unwrap())])
        } else if name.eq("Box") {
            let high = attribute.as_ref(py).getattr("high").unwrap();
            let high = high.extract::<PyReadonlyArrayDyn<f32>>().unwrap();
            let high = high.as_array();
            let low = attribute.as_ref(py).getattr("low").unwrap();
            let low = low.extract::<PyReadonlyArrayDyn<f32>>().unwrap();
            let low = low.as_array();

            let data: Vec<SpaceTypeBounds> = high
                .iter()
                .zip(low.iter())
                .map(|(h, l)| SpaceTypeBounds::Continuous(*l, *h))
                .collect();
            SpaceInfo::new(data)
        } else {
            println!("name is {name}");
            SpaceInfo::new(vec![SpaceTypeBounds::Discrete(2)])
        }
    }
}

impl DiscreteActionEnv for PyEnv {
    type Error = PyEnvErr;

    fn reset(&mut self) -> Result<ArrayD<f32>, PyEnvErr> {
        Python::with_gil(|py| -> Result<ArrayD<f32>, PyEnvErr> {
            // let kwargs = [("seed", 0)].into_py_dict(py);
            let Ok(call_result) = self.env.call_method(py, "reset", (), None) else {
                return Err(PyEnvErr::MethodNotFound);
            };
            let Ok(resulting_tuple) = call_result.downcast::<PyTuple>(py) else {
                return Err(PyEnvErr::NotExpectedReturn);
            };
            self.extract_state(resulting_tuple)
        })
    }

    fn step(&mut self, action: usize) -> Result<(ArrayD<f32>, f32, bool), PyEnvErr> {
        Python::with_gil(|py| -> Result<(ArrayD<f32>, f32, bool), PyEnvErr> {
            let Ok(call_result) =
                self.env
                    .call_method(py, "step", PyTuple::new(py, [action]), None)
            else {
                return Err(PyEnvErr::MethodNotFound);
            };
            let Ok(resulting_tuple) = call_result.downcast::<PyTuple>(py) else {
                return Err(PyEnvErr::NotExpectedReturn);
            };
            let state = self.extract_state(resulting_tuple)?;
            let reward = self.extract_reward(resulting_tuple)?;
            let terminated = self.extract_terminated(resulting_tuple)?;
            Ok((state, reward, terminated))
        })
    }

    fn render(&self) -> String {
        todo!()
    }

    fn observation_space(&self) -> SpaceInfo {
        Python::with_gil(|py| -> SpaceInfo {
            let attribute = self.env.getattr(py, "observation_space").unwrap();
            self.extract_space(attribute, py)
        })
    }

    fn action_space(&self) -> SpaceInfo {
        Python::with_gil(|py| -> SpaceInfo {
            let attribute = self.env.getattr(py, "action_space").unwrap();
            self.extract_space(attribute, py)
        })
    }
}

fn generate_policy(device: Device) -> (Box<dyn Module>, VarStore) {
    const NEURONS: i64 = 128;

    let mem_policy = nn::VarStore::new(device);
    let policy_net = nn::seq()
        .add(nn::linear(
            &mem_policy.root() / "al1",
            4,
            NEURONS,
            Default::default(),
        ))
        // .add_fn(|xs| xs.gelu("none"))
        .add_fn(|xs| xs.tanh())
        .add(nn::linear(
            &mem_policy.root() / "al2",
            NEURONS,
            2,
            Default::default(),
        ))
        .add_fn(|xs| xs.softmax(0, Kind::Float));
    (Box::new(policy_net), mem_policy)
}

#[pyfunction]
pub fn test(env: Py<PyAny>) -> PyResult<f32> {
    let mut rng: StdRng = StdRng::seed_from_u64(4);

    tch::manual_seed(rng.next_u64() as i64);
    tch::maybe_init_cuda();
    const MEM_SIZE: usize = 5_000;
    const MIN_MEM_SIZE: usize = 1_000;
    const GAMMA: f32 = 0.99;
    const UPDATE_FREQ: u128 = 10;
    const LEARNING_RATE: f64 = 0.0005;
    const EPSILON_DECAY: f32 = 0.0005;
    const START_EPSILON: f32 = 1.0;
    let device: Device = Device::Cpu;

    let train_env = PyEnv::new(env)?;

    let mem_replay = RandomExperienceBuffer::new(MEM_SIZE, MIN_MEM_SIZE, rng.next_u64(), device);

    let epsilon_greedy = EpsilonGreedy::new(
        START_EPSILON,
        rng.next_u64(),
        EpsilonUpdateStrategy::EpsilonDecreasing {
            final_epsilon: 0.0,
            epsilon_decay: Box::new(move |a| a - EPSILON_DECAY),
        },
    );

    let mut agent = DoubleDeepAgent::new(
        Box::new(epsilon_greedy),
        mem_replay,
        generate_policy,
        OptimizerEnum::Adam(nn::Adam::default()),
        LEARNING_RATE,
        GAMMA,
        device,
    );

    let mut trainer = CDTrainer::new(Box::new(train_env));
    trainer.early_stop = Some(Box::new(|reward| reward >= 500.0));

    let start = Instant::now();

    let r: Result<TrainResults, PyEnvErr> =
        trainer.train_by_steps2(&mut agent, 200_000, UPDATE_FREQ, 50, 10, false);
    let elapsed = start.elapsed();
    println!("Elapsed time: {elapsed:?}");
    let rewards = r.unwrap().3;
    let reward_max = rewards
        .iter()
        .fold(rewards[0], |o, r| if *r > o { *r } else { o });
    Ok(reward_max)
}
