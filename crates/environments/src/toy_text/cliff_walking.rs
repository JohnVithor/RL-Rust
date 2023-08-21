use std::ops::{Index, IndexMut};

use crate::{
    env::{DiscreteAction, DiscreteEnv, EnvError},
    utils::{from_2d_to_1d, inc},
};

#[derive(Debug, Copy, Clone)]
pub enum CliffWalkingAction {
    LEFT,
    DOWN,
    RIGHT,
    UP,
}

impl From<usize> for CliffWalkingAction {
    fn from(value: usize) -> Self {
        match value {
            0 => Self::LEFT,
            1 => Self::DOWN,
            2 => Self::RIGHT,
            3 => Self::UP,
            _ => panic!(),
        }
    }
}

impl Index<CliffWalkingAction> for [f64] {
    type Output = f64;

    fn index(&self, index: CliffWalkingAction) -> &Self::Output {
        &self[index as usize]
    }
}

impl IndexMut<CliffWalkingAction> for [f64] {
    fn index_mut(&mut self, index: CliffWalkingAction) -> &mut Self::Output {
        &mut self[index as usize]
    }
}

impl DiscreteAction for CliffWalkingAction {}

impl Index<CliffWalkingAction> for [(usize, f64, bool); 4] {
    type Output = (usize, f64, bool);

    fn index(&self, index: CliffWalkingAction) -> &Self::Output {
        match index {
            CliffWalkingAction::LEFT => &self[0],
            CliffWalkingAction::DOWN => &self[1],
            CliffWalkingAction::RIGHT => &self[2],
            CliffWalkingAction::UP => &self[3],
        }
    }
}

#[derive(Debug, Clone)]
pub struct CliffWalkingEnv {
    ready: bool,
    obs: [[(usize, f64, bool); 4]; 48],
    player_pos: usize,
    max_steps: u128,
    curr_step: u128,
}

impl CliffWalkingEnv {
    const START_POSITION: usize = 36;
    const CLIFF_POSITIONS: [usize; 10] = [37, 38, 39, 40, 41, 42, 43, 44, 45, 46];
    const GOAL_POSITION: usize = 47;
    const MAP: &str = "____________\n____________\n____________\n@!!!!!!!!!!G";

    fn update_probability_matrix(row: usize, col: usize, action: usize) -> (usize, f64, bool) {
        let (newrow, newcol) = inc(4, 12, row, col, action);
        let newstate: usize = from_2d_to_1d(12, newrow, newcol);
        let win: bool = newstate == Self::GOAL_POSITION;
        let lose: bool = Self::CLIFF_POSITIONS.contains(&newstate);
        let reward: f64 = if lose { -100.0 } else { -1.0 };
        (newstate, reward, lose || win)
    }

    pub fn new(max_steps: u128) -> Self {
        let ncol: usize = 12;
        let nrow: usize = 4;
        // calculating start positions probabilities
        let mut initial_state_distrib: [f64; 48] = [0.0; 48];
        initial_state_distrib[Self::START_POSITION] = 1.0;
        // calculating transitions probabilities
        let mut obs: [[(usize, f64, bool); 4]; 48] = [[(0, 0.0, false); 4]; 48];
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

impl DiscreteEnv<usize, CliffWalkingAction> for CliffWalkingEnv {
    fn reset(&mut self) -> usize {
        self.player_pos = Self::START_POSITION;
        self.ready = true;
        self.curr_step = 0;
        self.player_pos
    }

    fn step(&mut self, action: CliffWalkingAction) -> Result<(usize, f64, bool), EnvError> {
        if !self.ready {
            return Err(EnvError::EnvNotReady);
        }
        if self.curr_step >= self.max_steps {
            self.ready = false;
            return Ok((0, -100.0, true));
        }
        self.curr_step += 1;
        let obs: (usize, f64, bool) = self.obs[self.player_pos][action];
        self.player_pos = obs.0;
        if obs.2 {
            self.ready = false;
        }
        Ok(obs)
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
