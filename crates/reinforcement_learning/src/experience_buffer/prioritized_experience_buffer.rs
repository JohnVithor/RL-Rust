use std::collections::VecDeque;

use ndarray::Array1;
use rand::{rngs::StdRng, Rng, SeedableRng};
use tch::Tensor;

pub struct PrioritizedExperienceBuffer {
    capacity: usize,
    alpha: f64,
    priority_sum: Array1<f64>,
    priority_min: Array1<f64>,
    max_priority: f64,

    transitions: Array1<Transition>,
    next_idx: usize,
    size: usize,
    rng: StdRng,
}

impl PrioritizedExperienceBuffer {
    pub fn new(capacity: usize, alpha: f64, seed: u64) -> Self {
        Self {
            capacity,
            alpha,
            priority_sum: Array1::zeros(2 * capacity),
            priority_min: Array1::from_elem(2 * capacity, f64::INFINITY),
            max_priority: 1.0,
            transitions: Array1::default(capacity),
            next_idx: 0,
            size: 0,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub fn ready(&self) -> bool {
        self.transitions.len() == self.capacity
    }

    fn set_priority_min(&mut self, idx: usize, priority_alpha: f64) {
        // Leaf of the binary tree
        let mut idx = idx + self.capacity;
        self.priority_min[idx] = priority_alpha;

        // Update tree, by traversing along ancestors.
        // Continue until the root of the tree.
        while idx >= 2 {
            // Get the index of the parent node
            idx /= 2;
            // Value of the parent node is the minimum of it's two children
            self.priority_min[idx] = self.priority_min[2 * idx].min(self.priority_min[2 * idx + 1]);
        }
    }

    fn set_priority_sum(&mut self, idx: usize, priority: f64) {
        // Leaf of the binary tree
        let mut idx = idx + self.capacity;
        // Set the priority at the leaf
        self.priority_sum[idx] = priority;

        // Update tree, by traversing along ancestors.
        // Continue until the root of the tree.
        while idx >= 2 {
            // Get the index of the parent node
            idx /= 2;
            // Value of the parent node is the sum of it's two children
            self.priority_sum[idx] = self.priority_sum[2 * idx] + self.priority_sum[2 * idx + 1];
        }
    }
    pub fn add(&mut self, transition: Transition) {
        self.transitions[self.next_idx] = transition;
        self.size = self.capacity.min(self.size + 1);
        let priority_alpha = self.max_priority.powf(self.alpha);
        self.set_priority_min(self.next_idx, priority_alpha);
        self.set_priority_sum(self.next_idx, priority_alpha);
        self.next_idx = (self.next_idx + 1) % self.capacity;
    }

    pub fn find_prefix_sum_idx(&self, prefix_sum: f64) -> usize {
        // Start from the root
        let mut idx = 1;
        let mut prefix_sum = prefix_sum;
        while idx < self.capacity {
            // If the sum of the left branch is higher than required sum
            if self.priority_sum[idx * 2] > prefix_sum {
                // Go to left branch of the tree
                idx *= 2;
            } else {
                // Otherwise go to right branch and reduce the sum of left
                //  branch from required sum
                prefix_sum -= self.priority_sum[idx * 2];
                idx = 2 * idx + 1;
            }
        }
        // We are at the leaf node. Subtract the capacity by the index in the tree
        // to get the index of actual value
        idx - self.capacity
    }

    pub fn update_priorities(&mut self, indexes: &[usize], priorities: &[f64]) {
        for (idx, priority) in indexes.iter().zip(priorities.iter()) {
            // Set current max priority
            self.max_priority = self.max_priority.max(*priority);

            // Calculate $p_i^\alpha$
            let priority_alpha = priority.powf(self.alpha);
            // Update the trees
            self.set_priority_min(*idx, priority_alpha);
            self.set_priority_sum(*idx, priority_alpha);
        }
    }

    pub fn is_full(&self) -> bool {
        self.size == self.capacity
    }

    pub fn sample_batch(
        &mut self,
        size: usize,
    ) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) {
    }
}
