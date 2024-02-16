use std::cmp::{max, min};

use rand::{distributions::Uniform, prelude::Distribution};
use utils::categorical_sample;

use crate::{
    env::EnvError::EnvNotReady,
    space::{SpaceInfo, SpaceTypeBounds},
    utils::from_2d_to_1d,
    Env,
};

#[derive(Debug, Clone)]
pub struct TaxiEnv {
    ready: bool,
    initial_state_distrib: [f32; 500],
    obs: [[(usize, f32, bool); 6]; 500],
    curr_obs: usize,
    max_steps: u128,
    curr_step: u128,
}

impl TaxiEnv {
    const MAP: [&'static str; 7] = [
        "+---------+",
        "|R: | : :G|",
        "| : | : : |",
        "| : : : : |",
        "| | : | : |",
        "|Y| : |B: |",
        "+---------+",
    ];
    pub const LOCS: [(usize, usize); 4] = [(0, 0), (0, 4), (4, 0), (4, 3)];
    pub const ACTIONS: [&'static str; 6] = ["DOWN", "UP", "RIGHT", "LEFT", "PICKUP", "DROPOFF"];

    fn encode(taxi_row: usize, taxi_col: usize, pass_loc: usize, dest_loc: usize) -> usize {
        let mut i: usize = taxi_row;
        i *= 5;
        i += taxi_col;
        i *= 5;
        i += pass_loc;
        i *= 4;
        i += dest_loc;
        i
    }

    pub fn decode(i: usize) -> (usize, usize, usize, usize) {
        let mut out: (usize, usize, usize, usize) = (0, 0, 0, 0);
        let mut state = i;
        out.3 = state % 4;
        state /= 4;
        out.2 = state % 5;
        state /= 5;
        out.1 = state % 5;
        state /= 5;
        out.0 = state;
        out
    }

    pub fn new(max_steps: u128) -> Self {
        let mut initial_state_distrib: [f32; 500] = [0.0; 500];
        let mut obs: [[(usize, f32, bool); 6]; 500] = [[(0, 0.0, false); 6]; 500];
        let mut sum: f32 = 0.0;
        for row in 0..5 {
            for col in 0..5 {
                for pass_loc in 0..5 {
                    for dest_loc in 0..4 {
                        let state = Self::encode(row, col, pass_loc, dest_loc);
                        if pass_loc < 4 && pass_loc != dest_loc {
                            initial_state_distrib[state] += 1.0;
                            sum += 1.0;
                        }
                        for action in 0..6 {
                            let (mut new_row, mut new_col, mut new_pass_loc) = (row, col, pass_loc);
                            let mut reward = -1.0; // default reward when there is no pickup/dropoff
                            let mut terminated = false;
                            let taxi_loc = (row, col);

                            if action == 0 {
                                new_row = min(row + 1, 4)
                            } else if action == 1 {
                                new_row = if row != 0 { max(row - 1, 0) } else { 0 };
                            }
                            if action == 2
                                && Self::MAP[1 + row]
                                    .get((2 * col + 2)..(2 * col + 2) + 1)
                                    .unwrap()
                                    == ":"
                            {
                                new_col = min(col + 1, 4)
                            } else if action == 3
                                && Self::MAP[1 + row].get((2 * col)..(2 * col) + 1).unwrap() == ":"
                            {
                                new_col = if col != 0 { max(col - 1, 0) } else { 0 };
                            } else if action == 4 {
                                // pickup
                                if pass_loc < 4 && taxi_loc == Self::LOCS[pass_loc] {
                                    new_pass_loc = 4;
                                } else {
                                    // passenger not at location
                                    reward = -10.0;
                                }
                            } else if action == 5 {
                                // dropoff
                                if (taxi_loc == Self::LOCS[dest_loc]) && pass_loc == 4 {
                                    new_pass_loc = dest_loc;
                                    terminated = true;
                                    reward = 20.0;
                                } else {
                                    // dropoff at wrong location
                                    reward = -10.0;
                                }
                            }
                            let new_state = Self::encode(new_row, new_col, new_pass_loc, dest_loc);
                            obs[state][action] = (new_state, reward, terminated);
                        }
                    }
                }
            }
        }

        for value in &mut initial_state_distrib {
            *value /= sum;
        }

        Self {
            ready: false,
            initial_state_distrib,
            obs,
            curr_obs: 0,
            max_steps,
            curr_step: 0,
        }
    }
}

impl Env<usize, usize> for TaxiEnv {
    fn reset(&mut self) -> usize {
        let dist: Uniform<f32> = Uniform::from(0.0..1.0);
        let random: f32 = dist.sample(&mut rand::thread_rng());
        self.curr_obs = categorical_sample(self.initial_state_distrib.as_ref(), random);
        self.ready = true;
        self.curr_step = 0;
        self.curr_obs
    }

    fn step(&mut self, action: usize) -> Result<(usize, f32, bool), crate::EnvError> {
        if !self.ready {
            return Err(EnvNotReady);
        }
        if self.curr_step >= self.max_steps {
            self.ready = false;
            return Ok((0, 0.0, true));
        }
        self.curr_step += 1;
        let obs: (usize, f32, bool) = self.obs[self.curr_obs][action];
        self.curr_obs = obs.0;
        if obs.2 {
            self.ready = false;
        }
        Ok(obs)
    }

    fn render(&self) -> String {
        let mut new_map = Self::MAP.clone().join("\n");
        let (row, col, _, _) = Self::decode(self.curr_obs);
        let mut pos = from_2d_to_1d(11, row + 1, 2 * col + 1);
        for (i, _) in new_map.match_indices('\n') {
            if pos >= i {
                pos += 1;
            }
        }
        new_map.replace_range(pos..pos + 1, "T");
        new_map
    }

    fn observation_space(&self) -> SpaceInfo {
        SpaceInfo::new(vec![SpaceTypeBounds::Discrete(self.obs.len())])
    }

    fn action_space(&self) -> SpaceInfo {
        SpaceInfo::new(vec![SpaceTypeBounds::Discrete(6)])
    }
}
