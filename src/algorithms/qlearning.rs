use crate::algorithms::Agent;
use crate::algorithms::action_selection::ActionSelection;
use crate::env::{Observation, ActionSpace};
use crate::utils::argmax;
use std::cell::RefCell;

use super::Policy;
use super::policy_update::PolicyUpdate;

impl Agent {
    pub fn new(
        action_selection_strategy: Box<RefCell<dyn ActionSelection>>,
        policy_update_strategy: Box<RefCell<dyn PolicyUpdate>>,
        policy: Policy,
        discount_factor: f64,
        action_space: ActionSpace
    ) -> Self {
        return Self {
            action_selection_strategy,
            policy_update_strategy,
            discount_factor,
            policy,
            training_error: vec![],
            action_space
        };
    }
}

impl Agent for QLearningAgent {

    fn get_training_error(&self) -> &Vec<f64> {
        return &self.training_error;
    }

    fn get_action_selection_strategy(&self) -> &Box<RefCell<dyn ActionSelection>> {
        return &self.action_selection_strategy;
    }

    fn get_policy(&self) -> &Policy {
        return &self.policy;
    }

    fn get_action_space(&self) -> &ActionSpace {
        return &self.action_space;
    }

}