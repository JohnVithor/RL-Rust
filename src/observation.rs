use std::any::Any;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

pub trait Observation: Debug {
    fn base_size(&self) -> usize;
} 

impl Hash for dyn Observation {
    fn hash<H>(&self, state: &mut H) where H: Hasher {
        if self.base_size() > 0 {
            let ptr = self as *const dyn Observation as *const ();
            ptr.hash(state)
        } else {
            self.type_id().hash(state)
        }
    }
}

impl PartialEq for dyn Observation {
    fn eq(&self, other: &Self) -> bool {
        if self.type_id() == other.type_id() {
            let ptr1 = self as *const dyn Observation as *const ();
            let ptr2 = other as *const dyn Observation as *const ();
            self.base_size() == 0 || ptr1 == ptr2
        } else {
            false
        }
    }
}

impl Eq for dyn Observation {}

pub struct RcObs(Rc<dyn Observation>);

impl Hash for RcObs {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}