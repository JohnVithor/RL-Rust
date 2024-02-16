// use crate::action_selection::ActionSelection;
// use crate::model::Model;

// use super::{Agent, GetNextQValue};
// use std::cell::RefCell;
// use std::fmt::Debug;

// pub struct InternalModelAgent<'a, T: Clone + Debug, const COUNT: usize> {
//     agent: Box<RefCell<&'a mut dyn Agent<'a, T, COUNT>>>,
//     planning_steps: usize,
//     model: &'a mut dyn Model<T, COUNT>,
// }

// impl<'a, T: Clone + Debug, const COUNT: usize> InternalModelAgent<'a, T, COUNT> {
//     pub fn new(
//         agent: Box<RefCell<&'a mut dyn Agent<'a, T, COUNT>>>,
//         model: &'a mut dyn Model<T, COUNT>,
//         planing_length: usize,
//     ) -> Self {
//         Self {
//             agent,
//             model,
//             planning_steps: planing_length,
//         }
//     }
// }

// impl<'a, T: Clone + Debug, const COUNT: usize> Agent<'a, T, COUNT>
//     for InternalModelAgent<'a, T, COUNT>
// {
//     fn set_future_q_value_func(&mut self, func: GetNextQValue<COUNT>) {
//         self.agent.borrow_mut().set_future_q_value_func(func);
//     }

//     fn set_action_selector(&mut self, action_selector: &'a mut dyn ActionSelection<T, COUNT>) {
//         self.agent.borrow_mut().set_action_selector(action_selector);
//     }

//     fn get_action(&mut self, obs: &T) -> usize {
//         self.agent.borrow_mut().get_action(obs)
//     }

//     fn update(
//         &mut self,
//         curr_obs: &T,
//         curr_action: usize,
//         reward: f32,
//         terminated: bool,
//         next_obs: &T,
//         next_action: usize,
//     ) -> f32 {
//         let td = self.agent.borrow_mut().update(
//             curr_obs,
//             curr_action,
//             reward,
//             terminated,
//             next_obs,
//             next_action,
//         );
//         self.model.add_info(
//             curr_obs.clone(),
//             curr_action,
//             reward,
//             next_obs.clone(),
//             next_action,
//         );

//         for _i in 0..self.planning_steps {
//             let (curr_obs, curr_action, reward, next_obs, _next_action) = self.model.get_info();
//             let next_action = self.agent.borrow_mut().get_action(&next_obs);
//             self.agent.borrow_mut().update(
//                 &curr_obs,
//                 curr_action,
//                 reward,
//                 false,
//                 &next_obs,
//                 next_action,
//             );
//         }
//         td
//     }

//     fn reset(&mut self) {
//         self.agent.borrow_mut().reset();
//         self.model.reset();
//     }
// }
