use std::cell::{RefCell, RefMut};
use std::hash::Hash;

use crate::env::ActionSpace;
use crate::policy::Policy;
use crate::utils::{argmax, max};

use super::action_selection::ActionSelection;

pub type NextQvalueFunction<T> = fn (T, usize, T, usize, RefMut<'_, &mut dyn Policy<T>>, &Box<RefCell<&mut dyn ActionSelection<T>>>) -> f64;


pub fn sarsa<T: Hash+PartialEq+Eq+Clone>(
    _curr_obs: T,
    _curr_action: usize,
    next_obs: T,
    next_action: usize,
    mut policy: RefMut<'_, &mut dyn Policy<T>>,
    _action_selection: &Box<RefCell<&mut dyn ActionSelection<T>>>
) -> f64 {
    return policy.get_values(next_obs)[next_action];
}

pub fn q_learning<T: Hash+PartialEq+Eq+Clone>(
    _curr_obs: T,
    _curr_action: usize,
    next_obs: T,
    _next_action: usize,
    mut policy: RefMut<'_, &mut dyn Policy<T>>,
    _action_selection: &Box<RefCell<&mut dyn ActionSelection<T>>>
) -> f64 {
    return max(policy.get_values(next_obs));
}

pub fn expected_sarsa<T: Hash+PartialEq+Eq+Clone>(
    _curr_obs: T,
    _curr_action: usize,
    next_obs: T,
    _next_action: usize,
    mut policy: RefMut<'_, &mut dyn Policy<T>>,
    action_selection: &Box<RefCell<&mut dyn ActionSelection<T>>>
) -> f64 {
    let action_space: ActionSpace = policy.get_action_space().clone();
    let next_q_values: &ndarray::Array1<f64> = policy.get_values(next_obs.clone());
    let epsilon: f64 = action_selection.borrow().get_exploration_rate();
    let policy_probs: ndarray::Array1<f64> = action_selection.borrow().get_exploration_probs(&action_space);
    let best_action_value: f64 = next_q_values[argmax(&next_q_values)];
    
    let mut n_max_action: i32 = 0;
    for i in 0..action_space.size {
        if next_q_values[i] == best_action_value {
            n_max_action += 1;
        }
    }
    let mut future_q_value: f64 = 0.0;
    for i in 0..action_space.size {
        if next_q_values[i] == best_action_value {
            future_q_value += (policy_probs[i] + (1.0-epsilon) / n_max_action as f64) * next_q_values[i]
        } else {
            future_q_value += policy_probs[i] * next_q_values[i]
        }
    }
    return future_q_value;
}