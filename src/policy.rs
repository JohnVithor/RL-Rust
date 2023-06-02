mod basic_policy;
mod double_policy;

pub use basic_policy::BasicPolicy;
pub use double_policy::DoublePolicy;
use std::{hash::Hash, rc::Rc};
use fxhash::FxHashMap;

use crate::{env::{ActionSpace}, observation::Observation};
pub trait Policy{
    fn get_action_space(&self) -> &ActionSpace;

    fn get_ref(&mut self, curr_obs: Rc<dyn Observation>) -> &Vec<f64>;

    fn get_ref_if_has(&mut self, curr_obs: &Rc<dyn Observation>) -> Option<&Vec<f64>>;

    fn get_mut(&mut self, curr_obs: Rc<dyn Observation>) -> &mut Vec<f64>;

    fn get_mut_if_has(&mut self, curr_obs: &Rc<dyn Observation>) -> Option<&mut Vec<f64>>;

    fn get_mut_values(&mut self) -> &mut FxHashMap<Rc<dyn Observation>, Vec<f64>>;

    fn reset(&mut self);

    fn after_update(&mut self);
}
