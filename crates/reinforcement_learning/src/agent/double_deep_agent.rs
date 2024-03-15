use crate::{
    action_selection::ContinuousObsDiscreteActionSelection,
    agent::ContinuousObsDiscreteActionAgent, experience_buffer::RandomExperienceBuffer,
};
use ndarray::Array1;
use tch::{
    nn::{Adam, AdamW, Module, Optimizer, OptimizerConfig, RmsProp, Sgd, VarStore},
    COptimizer, Device, Kind, TchError, Tensor,
};

pub enum OptimizerEnum {
    Adam(Adam),
    Sgd(Sgd),
    RmsProp(RmsProp),
    AdamW(AdamW),
}

impl OptimizerConfig for OptimizerEnum {
    fn build_copt(&self, lr: f64) -> Result<COptimizer, TchError> {
        match self {
            OptimizerEnum::Adam(opt) => opt.build_copt(lr),
            OptimizerEnum::Sgd(opt) => opt.build_copt(lr),
            OptimizerEnum::RmsProp(opt) => opt.build_copt(lr),
            OptimizerEnum::AdamW(opt) => opt.build_copt(lr),
        }
    }
}

pub struct DoubleDeepAgent {
    pub action_selection: Box<dyn ContinuousObsDiscreteActionSelection>,
    pub policy: Box<dyn Module>,
    pub target_policy: Box<dyn Module>,
    pub policy_vs: VarStore,
    pub target_policy_vs: VarStore,
    pub optimizer: Optimizer,
    pub memory: RandomExperienceBuffer,
    pub discount_factor: f32,
}

impl DoubleDeepAgent {
    pub fn new(
        action_selector: Box<dyn ContinuousObsDiscreteActionSelection>,
        mem_replay: RandomExperienceBuffer,
        generate_policy: fn(device: Device) -> (Box<dyn Module>, VarStore),
        opt: OptimizerEnum,
        lr: f64,
        discount_factor: f32,
        device: Device,
    ) -> Self {
        let (policy_net, mem_policy) = generate_policy(device);
        let (target_net, mut mem_target) = generate_policy(device);
        mem_target.copy(&mem_policy).unwrap();
        Self {
            optimizer: opt.build(&mem_policy, lr).unwrap(),
            action_selection: action_selector,
            memory: mem_replay,
            policy: policy_net,
            policy_vs: mem_policy,
            target_policy: target_net,
            target_policy_vs: mem_target,
            discount_factor,
        }
    }

    pub fn memorize(
        &mut self,
        curr_obs: &Array1<f32>,
        curr_action: usize,
        reward: f32,
        terminated: bool,
        next_obs: &Array1<f32>,
        next_action: usize,
    ) {
        let curr_state = Tensor::try_from(curr_obs).unwrap();
        let next_state = Tensor::try_from(next_obs).unwrap();

        self.memory.add(
            &curr_state,
            curr_action,
            reward,
            terminated,
            &next_state,
            next_action,
        );
    }

    pub fn optimize(&mut self, loss: Tensor) {
        self.optimizer.zero_grad();
        loss.backward();
        self.optimizer.step();
    }
}

impl ContinuousObsDiscreteActionAgent for DoubleDeepAgent {
    fn get_action(&mut self, state: &Tensor) -> usize {
        let values = tch::no_grad(|| self.policy.forward(state));
        let values: ndarray::ArrayD<f32> = (&values).try_into().unwrap();
        let len = values.len();
        self.action_selection
            .get_action(&values.into_shape(len).unwrap()) as usize
    }

    fn get_best_action(&self, state: &Tensor) -> usize {
        let values = tch::no_grad(|| self.policy.forward(state));
        let a: i32 = values.argmax(0, true).try_into().unwrap();
        a as usize
    }

    fn add_transition(
        &mut self,
        curr_state: &Tensor,
        curr_action: usize,
        reward: f32,
        done: bool,
        next_state: &Tensor,
        next_action: usize,
    ) {
        self.memory.add(
            curr_state,
            curr_action,
            reward,
            done,
            next_state,
            next_action,
        );
    }

    fn update_networks(&mut self) -> Result<(), TchError> {
        self.target_policy_vs.copy(&self.policy_vs)
    }

    fn get_batch(&mut self, size: usize) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) {
        self.memory.sample_batch(size)
    }

    fn batch_qvalues(&self, b_states: &Tensor, b_actions: &Tensor) -> Tensor {
        self.policy.forward(b_states).gather(1, b_actions, false)
    }

    fn batch_expected_values(
        &self,
        b_state_: &Tensor,
        b_reward: &Tensor,
        b_done: &Tensor,
    ) -> Tensor {
        let best_target_qvalues =
            tch::no_grad(|| self.target_policy.forward(b_state_).max_dim(1, true).0);
        b_reward + self.discount_factor * (&Tensor::from(1.0) - b_done) * (&best_target_qvalues)
    }

    fn optimize(&mut self, loss: &Tensor) {
        self.optimizer.zero_grad();
        loss.backward();
        self.optimizer.step();
    }

    fn reset(&mut self) {
        self.action_selection.reset();
        // TODO: reset policies
    }

    fn update(&mut self) -> Option<f32> {
        if self.memory.ready() {
            let (b_state, b_action, b_reward, b_done, b_state_, _) = self.get_batch(32);
            let policy_qvalues = self.batch_qvalues(&b_state, &b_action);
            let expected_values = self.batch_expected_values(&b_state_, &b_reward, &b_done);
            let loss = policy_qvalues.mse_loss(&expected_values, tch::Reduction::Mean);
            self.optimize(loss);
            Some(expected_values.mean(Kind::Float).try_into().unwrap())
        } else {
            None
        }
    }
    fn episode_end_hook(&mut self) {
        // self.action_selection.update(0.0);
    }

    fn action_selection_update(&mut self, epi_reward: f32) {
        self.action_selection.update(epi_reward);
    }
}
