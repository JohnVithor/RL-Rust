
use rand::distributions::{Distribution, Uniform};

#[derive(Debug, Clone)]
pub struct ActionSpace {
    pub size: usize,
    dist: Uniform<usize>
}

impl ActionSpace {
    pub fn new(size: usize) -> Self {
        return Self {
            size,
            dist: Uniform::from(0..size)
        }
    }

    pub fn sample(&self) -> usize {
        return self.dist.sample(&mut rand::thread_rng());
    }
}