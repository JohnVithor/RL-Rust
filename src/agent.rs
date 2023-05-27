use std::cell::RefCell;
use std::hash::Hash;
use crate::env::ActionSpace;
use crate::algorithms::action_selection::ActionSelection;
use crate::algorithms::policy_update::PolicyUpdate;
use crate::policy::Policy;

pub struct Agent<T> {
    action_selection_strategy: Box<RefCell<dyn ActionSelection<T>>>,
    policy_update_strategy: Box<RefCell<dyn PolicyUpdate<T>>>,
    pub policy: Policy<T>,
    training_error: Vec<f64>,
    action_space: ActionSpace
}

impl<T: Hash+PartialEq+Eq+Clone> Agent<T> {
    pub fn new(
        action_selection_strategy: Box<RefCell<dyn ActionSelection<T>>>,
        policy_update_strategy: Box<RefCell<dyn PolicyUpdate<T>>>,
        policy: Policy<T>,
        action_space: ActionSpace
    ) -> Self {
        return Self {
            action_selection_strategy,
            policy_update_strategy,
            policy,
            training_error: vec![],
            action_space
        };
    }
    pub fn get_action(&mut self, obs:&T) -> usize {
        return self.action_selection_strategy.borrow_mut().get_action(obs, &self.action_space, &self.policy); 
    }

    pub fn get_training_error(&self) -> &Vec<f64> {
        return &self.training_error;
    }

    pub fn get_action_selection_strategy(&self) -> &Box<RefCell<dyn ActionSelection<T>>> {
        return &self.action_selection_strategy;
    }

    pub fn get_policy(&self) -> &Policy<T> {
        return &self.policy;
    }

    pub fn get_action_space(&self) -> &ActionSpace {
        return &self.action_space;
    }

    pub fn update(
        &mut self,
        curr_obs: &T,
        curr_action: usize,
        reward: f64,
        terminated: bool,
        next_obs: &T,
        next_action: usize
    ) {
        if terminated {
            self.action_selection_strategy.borrow_mut().update();
        }
        self.policy_update_strategy.borrow_mut().update(
            curr_obs.clone(),
            curr_action,
            next_obs.clone(),
            next_action,
            reward,
            terminated,
            &mut self.policy,
            &self.action_selection_strategy
        );

        // self.training_error.push(temporal_difference);
    }
}