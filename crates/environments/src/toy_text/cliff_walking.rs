use ndarray::{Array1, ArrayD};

use crate::{
    env::DiscreteActionEnv,
    space::{SpaceInfo, SpaceTypeBounds},
    utils::{from_2d_to_1d, inc},
};

pub enum CliffWalkingError {
    NotReady,
    InvalidAction,
}

#[derive(Debug, Clone)]
pub struct CliffWalkingEnv {
    ready: bool,
    obs: [[(usize, f32, bool); 4]; 48],
    player_pos: usize,
    max_steps: u128,
    curr_step: u128,
}

impl CliffWalkingEnv {
    const START_POSITION: usize = 36;
    const CLIFF_POSITIONS: [usize; 10] = [37, 38, 39, 40, 41, 42, 43, 44, 45, 46];
    const GOAL_POSITION: usize = 47;
    const MAP: &'static str = "____________\n____________\n____________\n@!!!!!!!!!!G";

    fn update_probability_matrix(row: usize, col: usize, action: usize) -> (usize, f32, bool) {
        let (newrow, newcol) = inc(4, 12, row, col, action);
        let newstate: usize = from_2d_to_1d(12, newrow, newcol);
        let win: bool = newstate == Self::GOAL_POSITION;
        let lose: bool = Self::CLIFF_POSITIONS.contains(&newstate);
        let reward: f32 = if lose { -100.0 } else { -1.0 };
        (newstate, reward, lose || win)
    }

    pub fn new(max_steps: u128) -> Self {
        let ncol: usize = 12;
        let nrow: usize = 4;
        // calculating start positions probabilities
        let mut initial_state_distrib: [f32; 48] = [0.0; 48];
        initial_state_distrib[Self::START_POSITION] = 1.0;
        // calculating transitions probabilities
        let mut obs: [[(usize, f32, bool); 4]; 48] = [[(0, 0.0, false); 4]; 48];
        for row in 0..nrow {
            for col in 0..ncol {
                for a in 0..4 {
                    let i = from_2d_to_1d(ncol, row, col);
                    obs[i][a] = Self::update_probability_matrix(row, col, a);
                }
            }
        }

        let env: CliffWalkingEnv = Self {
            ready: false,
            obs,
            player_pos: 0,
            max_steps,
            curr_step: 0,
        };
        env
    }
}

impl DiscreteActionEnv for CliffWalkingEnv {
    type Error = CliffWalkingError;
    fn reset(&mut self) -> Result<ArrayD<f32>, CliffWalkingError> {
        self.player_pos = Self::START_POSITION;
        self.ready = true;
        self.curr_step = 0;
        Ok(Array1::from_elem(1, self.player_pos as f32).into_dyn())
    }

    fn observation_space(&self) -> SpaceInfo {
        SpaceInfo::new(vec![SpaceTypeBounds::Discrete(48)])
    }

    fn action_space(&self) -> SpaceInfo {
        SpaceInfo::new(vec![SpaceTypeBounds::Discrete(4)])
    }

    fn step(&mut self, action: usize) -> Result<(ArrayD<f32>, f32, bool), CliffWalkingError> {
        if !self.ready {
            return Err(CliffWalkingError::NotReady);
        }
        if self.curr_step >= self.max_steps {
            self.ready = false;
            let state = Array1::from_elem(1, 0.0).into_dyn();
            return Ok((state, -100.0, true));
        }
        self.curr_step += 1;
        let (state, reward, terminated): (usize, f32, bool) = self.obs[self.player_pos][action];
        self.player_pos = state;
        if terminated {
            self.ready = false;
        }
        let state = Array1::from_elem(1, state as f32).into_dyn();
        Ok((state, reward, terminated))
    }

    fn render(&self) -> String {
        let mut new_map: String = <&str>::clone(&Self::MAP).to_string();
        new_map.replace_range(39..40, "_");
        let mut pos: usize = self.player_pos;
        for (i, _) in new_map.match_indices('\n') {
            if pos >= i {
                pos += 1;
            }
        }
        new_map.replace_range(pos..pos + 1, "@");
        new_map
    }
}
