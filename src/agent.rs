use std::cell::RefCell;
use std::hash::Hash;
use std::fmt::Debug;
use crate::env::{ActionSpace, Env};
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

impl<T: Hash+PartialEq+Eq+Clone+Debug> Agent<T> {
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
        let temporal_difference: f64 = self.policy_update_strategy.borrow_mut().update(
            curr_obs.clone(),
            curr_action,
            next_obs.clone(),
            next_action,
            reward,
            terminated,
            &mut self.policy,
            &self.action_selection_strategy
        );

        self.training_error.push(temporal_difference);
    }

    pub fn train(&mut self, env: &mut dyn Env<T>, n_episodes: u128) -> (Vec<f64>, Vec<u128>) {
        let mut reward_history: Vec<f64> = vec![];
        let mut episode_length: Vec<u128> = vec![];
        for _episode in 0..n_episodes {
            let mut action_counter: u128 = 0;
            let mut epi_reward: f64 = 0.0;
            let mut curr_obs: T = env.reset();
            let mut curr_action: usize = self.get_action(&curr_obs);
            loop {
                action_counter+=1;
                let (next_obs, reward, terminated) = env.step(curr_action).unwrap();
                let next_action: usize = self.get_action(&next_obs);
                self.update(
                    &curr_obs,
                    curr_action,
                    reward,
                    terminated,
                    &next_obs,
                    next_action,
                );
                curr_obs = next_obs;
                curr_action = next_action;
                epi_reward += reward;
                if terminated {
                    reward_history.push(epi_reward);
                    break;
                }
            }
            episode_length.push(action_counter);
        }
        return (reward_history, episode_length);
    }
}