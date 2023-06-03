use std::cell::RefCell;
use std::hash::Hash;
use std::fmt::Debug;
use crate::algorithms::policy_update::NextQvalueFunction;
use crate::env::{ActionSpace, Env};
use crate::algorithms::action_selection::ActionSelection;
use crate::policy::Policy;

pub struct Agent<'a, T> {
    action_selection_strategy: Box<RefCell<&'a mut dyn ActionSelection<T>>>,
    next_qvalue_function: NextQvalueFunction<T>,
    policy: Box<RefCell<&'a mut dyn Policy<T>>>,
    training_error: Vec<f64>,
    action_space: ActionSpace
}

impl<'a, T: Hash+PartialEq+Eq+Clone+Debug> Agent<'a, T> {
    pub fn new(
        action_selection_strategy: &'a mut (dyn ActionSelection<T> + 'a),
        next_qvalue_function: NextQvalueFunction<T>,
        policy: &'a mut (dyn Policy<T> + 'a),
        action_space: ActionSpace
    ) -> Self {
        action_selection_strategy.reset();
        policy.reset();
        return Self {
            action_selection_strategy: Box::new(RefCell::new(action_selection_strategy)),
            next_qvalue_function,
            policy: Box::new(RefCell::new(policy)),
            training_error: vec![],
            action_space
        };
    }
    pub fn get_action(&mut self, obs:&T) -> usize {
        return self.action_selection_strategy.borrow_mut().get_action(obs,  &mut self.policy.borrow_mut()); 
    }

    pub fn get_training_error(&self) -> &Vec<f64> {
        return &self.training_error;
    }

    pub fn get_action_selection_strategy(&self) -> &Box<RefCell<&'a mut dyn ActionSelection<T>>> {
        return &self.action_selection_strategy;
    }

    pub fn get_policy(&self) -> &Box<RefCell<&'a mut (dyn Policy<T>)>> {
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
        let future_q_value: f64 = (self.next_qvalue_function)(
            curr_obs.clone(),
            curr_action,
            next_obs.clone(),
            next_action,
            self.policy.borrow_mut(),
            &self.action_selection_strategy
        );
        let temporal_difference: f64 = self.policy.borrow_mut().get_td(curr_obs.clone(), curr_action, reward, future_q_value);
        self.policy.borrow_mut().update_values(curr_obs.clone(), curr_action, next_obs.clone(), next_action, temporal_difference);
        self.policy.borrow_mut().after_update();
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

    pub fn example(&mut self, env: &mut dyn Env<T>) {
        let mut epi_reward = 0.0;
        let mut curr_action: usize = self.get_action(&env.reset());
        let mut steps: i32 = 0;
        for _i in 0..100 {
            steps+=1;
            println!("{}", env.render());
            let (next_obs, reward, terminated) = env.step(curr_action).unwrap();
            let next_action: usize = self.get_action(&next_obs);
            println!("{:?}", env.get_action_label(curr_action));
            println!("step reward {:?}", reward);
            curr_action = next_action;
            epi_reward+=reward;
            if terminated {
                println!("{}", env.render());
                println!("episode reward {:?}", epi_reward);
                println!("terminated with {:?} steps", steps);
                break;
            }
        }
    }
}