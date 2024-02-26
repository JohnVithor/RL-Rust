use ndarray::Array1;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use tch::Tensor;

pub struct PrioritizedExperienceBuffer {
    capacity: usize,
    alpha: f64,
    priority_sum: Array1<f64>,
    priority_min: Array1<f64>,
    max_priority: f64,
    curr_states: Array1<Tensor>,
    curr_actions: Array1<Tensor>,
    rewards: Array1<Tensor>,
    next_states: Array1<Tensor>,
    next_actions: Array1<Tensor>,
    dones: Array1<Tensor>,
    next_idx: usize,
    size: usize,
    rng: SmallRng,
}

impl PrioritizedExperienceBuffer {
    pub fn new(capacity: usize, alpha: f64, seed: u64) -> Self {
        Self {
            capacity,
            alpha,
            priority_sum: Array1::zeros(2 * capacity),
            priority_min: Array1::from_elem(2 * capacity, f64::INFINITY),
            max_priority: 1.0,
            curr_states: Array1::default(capacity),
            curr_actions: Array1::default(capacity),
            rewards: Array1::default(capacity),
            next_states: Array1::default(capacity),
            next_actions: Array1::default(capacity),
            dones: Array1::default(capacity),
            next_idx: 0,
            size: 0,
            rng: SmallRng::seed_from_u64(seed),
        }
    }

    pub fn ready(&self) -> bool {
        self.size == self.capacity
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
    pub fn add(
        &mut self,
        curr_state: &Tensor,
        curr_action: i64,
        reward: f32,
        done: bool,
        next_state: &Tensor,
        next_action: i64,
    ) {
        self.curr_states[self.next_idx] = curr_state.shallow_clone();
        self.curr_actions[self.next_idx] = Tensor::from(curr_action);
        self.rewards[self.next_idx] = Tensor::from(reward);
        self.dones[self.next_idx] = Tensor::from(done as i64);
        self.next_states[self.next_idx] = next_state.shallow_clone();
        self.next_actions[self.next_idx] = Tensor::from(next_action);
        let priority_alpha = self.max_priority.powf(self.alpha);
        self.set_priority_min(self.next_idx, priority_alpha);
        self.set_priority_sum(self.next_idx, priority_alpha);
        self.next_idx = (self.next_idx + 1) % self.capacity;
        self.size = self.capacity.min(self.size + 1);
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
        beta: f64,
    ) -> (
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ) {
        let mut indexes = Array1::zeros(size);
        let mut weights = Array1::zeros(size);
        for i in 0..size {
            let p = self.rng.gen_range(0.0..1.0) * self.priority_sum[1];
            let idx = self.find_prefix_sum_idx(p);
            indexes[i] = idx;
        }
        let prob_min = self.priority_min[1] / self.priority_sum[1];
        let max_weight = (prob_min * self.size as f64).powf(-beta);

        for i in 0..size {
            let idx = indexes[i];
            let prob = self.priority_sum[idx + self.capacity] / self.priority_sum[1];
            let weight = (prob * self.size as f64).powf(-beta);
            weights[i] = weight / max_weight;
        }
        let mut curr_obs: Vec<Tensor> = Vec::new();
        let mut curr_actions: Vec<Tensor> = Vec::new();
        let mut rewards: Vec<Tensor> = Vec::new();
        let mut dones: Vec<Tensor> = Vec::new();
        let mut next_obs: Vec<Tensor> = Vec::new();
        let mut next_actions: Vec<Tensor> = Vec::new();
        indexes.iter().for_each(|i| {
            curr_obs.push(self.curr_states[*i].shallow_clone());
            curr_actions.push(self.curr_actions[*i].shallow_clone());
            rewards.push(self.rewards[*i].shallow_clone());
            dones.push(self.dones[*i].shallow_clone());
            next_obs.push(self.next_states[*i].shallow_clone());
            next_actions.push(self.next_actions[*i].shallow_clone());
        });
        let indexes: Vec<i64> = indexes.iter().map(|x| *x as i64).collect();
        (
            Tensor::stack(&curr_obs, 0),
            Tensor::stack(&curr_actions, 0).reshape([-1, 1]),
            Tensor::stack(&rewards, 0).reshape([-1, 1]),
            Tensor::stack(&dones, 0).reshape([-1, 1]),
            Tensor::stack(&next_obs, 0),
            Tensor::stack(&next_actions, 0).reshape([-1, 1]),
            Tensor::from_slice(&indexes).reshape([-1, 1]),
            Tensor::from_slice(weights.as_slice().unwrap()).reshape([-1, 1]),
        )
    }
}
