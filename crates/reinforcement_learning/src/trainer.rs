use environments::env::DiscreteEnv;

use crate::agent::DiscreteAgent;
pub type TrainResults = (Vec<f64>, Vec<u128>, Vec<f64>, Vec<f64>, Vec<f64>);

pub struct DiscreteTrainer<Obs, Action> {
    // action_to_repr: Rc<dyn Fn(&Action) -> usize>,
    pub repr_to_action: fn(usize) -> Action,
    pub obs_to_repr: fn(&Obs) -> usize,
    // repr_to_obs: Rc<dyn Fn(usize) -> Obs>,
}

impl<Obs, Action> DiscreteTrainer<Obs, Action> {
    pub fn new(repr_to_action: fn(usize) -> Action, obs_to_repr: fn(&Obs) -> usize) -> Self {
        Self {
            repr_to_action,
            obs_to_repr,
        }
    }

    pub fn train(
        &mut self,
        env: &mut dyn DiscreteEnv<Obs, Action>,
        agent: &mut dyn DiscreteAgent,
        n_episodes: u128,
        eval_at: u128,
        eval_for: u128,
    ) -> TrainResults
    where
        Obs: Clone,
    {
        let mut training_reward: Vec<f64> = vec![];
        let mut training_length: Vec<u128> = vec![];
        let mut training_error: Vec<f64> = vec![];
        let mut evaluation_reward: Vec<f64> = vec![];
        let mut evaluation_length: Vec<f64> = vec![];

        for episode in 0..n_episodes {
            let mut action_counter: u128 = 0;
            let mut epi_reward: f64 = 0.0;
            let mut curr_obs: Obs = env.reset();
            let mut curr_action_repr: usize = agent.get_action((self.obs_to_repr)(&curr_obs));

            loop {
                action_counter += 1;
                let curr_action = (self.repr_to_action)(curr_action_repr);
                let (next_obs, reward, terminated) = env.step(curr_action).unwrap();
                let next_obs_repr = (self.obs_to_repr)(&next_obs);
                let next_action_repr: usize = agent.get_action(next_obs_repr);
                let td = agent.update(
                    (self.obs_to_repr)(&curr_obs),
                    curr_action_repr,
                    reward,
                    terminated,
                    next_obs_repr,
                    next_action_repr,
                );
                training_error.push(td);
                curr_obs = next_obs;
                curr_action_repr = next_action_repr;
                epi_reward += reward;
                if terminated {
                    training_reward.push(epi_reward);
                    break;
                }
            }
            if episode % eval_at == 0 {
                let (r, l) = self.evaluate(env, agent, eval_for);
                let mr: f64 = r.iter().sum::<f64>() / r.len() as f64;
                let ml: f64 = l.iter().sum::<u128>() as f64 / l.len() as f64;
                evaluation_reward.push(mr);
                evaluation_length.push(ml);
            }
            training_length.push(action_counter);
        }
        (
            training_reward,
            training_length,
            training_error,
            evaluation_reward,
            evaluation_length,
        )
    }

    pub fn evaluate(
        &self,
        env: &mut dyn DiscreteEnv<Obs, Action>,
        agent: &mut dyn DiscreteAgent,
        n_episodes: u128,
    ) -> (Vec<f64>, Vec<u128>) {
        let mut reward_history: Vec<f64> = vec![];
        let mut episode_length: Vec<u128> = vec![];
        for _episode in 0..n_episodes {
            let mut action_counter: u128 = 0;
            let mut epi_reward: f64 = 0.0;
            let obs_repr = (self.obs_to_repr)(&env.reset());
            let action_repr: usize = agent.get_action(obs_repr);
            let mut curr_action = (self.repr_to_action)(action_repr);
            loop {
                action_counter += 1;
                let (obs, reward, terminated) = env.step(curr_action).unwrap();
                let next_obs_repr = (self.obs_to_repr)(&obs);
                let next_action_repr: usize = agent.get_action(next_obs_repr);
                let next_action = (self.repr_to_action)(next_action_repr);
                curr_action = next_action;
                epi_reward += reward;
                if terminated {
                    reward_history.push(epi_reward);
                    break;
                }
            }
            episode_length.push(action_counter);
        }
        (reward_history, episode_length)
    }
}
