use crate::env::{Env, EnvNotReady};
use crate::utils::{categorical_sample, from_1d_to_2d, from_2d_to_1d, inc};

use rand::distributions::Uniform;
use rand::prelude::Distribution;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum FrozenLakeTerrain {
    START,
    WALL,
    HOLE,
    #[default]
    GROUND,
    GOAL,
}

impl FrozenLakeTerrain {
    pub fn value(&self) -> f64 {
        match self {
            FrozenLakeTerrain::HOLE => -1.0,
            FrozenLakeTerrain::WALL => -0.5,
            FrozenLakeTerrain::START => 0.0,
            FrozenLakeTerrain::GROUND => 0.5,
            FrozenLakeTerrain::GOAL => 1.0,
        }
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, Default)]
pub struct FrozenLakeObs {
    pub left: FrozenLakeTerrain,
    pub down: FrozenLakeTerrain,
    pub right: FrozenLakeTerrain,
    pub up: FrozenLakeTerrain,
    pub x: usize,
    pub y: usize,
}

impl FrozenLakeObs {
    pub fn get(&self, action: usize) -> FrozenLakeTerrain {
        match action {
            0 => self.left,
            1 => self.down,
            2 => self.right,
            3 => self.up,
            _ => panic!("Invalid Action!"),
        }
    }
}

type Transition = (f64, FrozenLakeObs, usize, f64, bool);

#[derive(Debug, Clone)]
pub struct FrozenLakePartialEnv {
    ready: bool,
    initial_state_distrib: Vec<f64>,
    probs: Vec<[[Transition; 3]; 4]>,
    player_pos: usize,
    dist: Uniform<f64>,
    max_steps: u128,
    curr_step: u128,
    map: Vec<String>,
    ncol: usize,
}

impl FrozenLakePartialEnv {
    // default 4x4 map
    pub const MAP_4X4: [&str; 4] = ["SFFF", "FHFH", "FFFH", "HFFG"];

    pub const MAP_8X8: [&str; 8] = [
        "SFFFFFFF", "FFFFFFFF", "FFFHFFFF", "FFFFFHFF", "FFFHFFFF", "FHHFFFHF", "FHFFHFHF",
        "FFFHFFFG",
    ];

    pub const ACTIONS: [&str; 4] = ["LEFT", "DOWN", "RIGHT", "UP"];

    fn update_probability_matrix(
        map: &Vec<String>,
        nrow: usize,
        ncol: usize,
        row: usize,
        col: usize,
        action: usize,
    ) -> (FrozenLakeObs, usize, f64, bool) {
        let curr_obs = Self::get_obs(map, row, col);
        let next_terrain = curr_obs.get(action);

        let (new_row, new_col) = inc(nrow, ncol, row, col, action);
        let new_obs = Self::get_obs(map, new_row, new_col);
        let p_pos = from_2d_to_1d(ncol, new_row, new_col);

        let win = next_terrain == FrozenLakeTerrain::GOAL;
        let terminated: bool = win || next_terrain == FrozenLakeTerrain::HOLE;
        let reward: f64 = if win { 10.0 } else { -1.0 };
        (new_obs, p_pos, reward, terminated)
    }

    fn get_obs(map: &Vec<String>, row: usize, col: usize) -> FrozenLakeObs {
        let nrow: usize = map.len();
        let ncol: usize = map[0].len();
        let left_pos = (row, col - 1);
        let down_pos = (row + 1, col);
        let right_pos = (row, col + 1);
        let up_pos = (row - 1, col);
        FrozenLakeObs {
            left: if col == 0 {
                FrozenLakeTerrain::WALL
            } else {
                FrozenLakePartialEnv::get_terrain(map, left_pos.0, left_pos.1)
            },
            down: if row == nrow - 1 {
                FrozenLakeTerrain::WALL
            } else {
                FrozenLakePartialEnv::get_terrain(map, down_pos.0, down_pos.1)
            },
            right: if col == ncol - 1 {
                FrozenLakeTerrain::WALL
            } else {
                FrozenLakePartialEnv::get_terrain(map, right_pos.0, right_pos.1)
            },
            up: if row == 0 {
                FrozenLakeTerrain::WALL
            } else {
                FrozenLakePartialEnv::get_terrain(map, up_pos.0, up_pos.1)
            },
            x: row,
            y: col,
        }
    }

    fn get_terrain(map: &[String], row: usize, col: usize) -> FrozenLakeTerrain {
        let row = map.get(row);
        return match row {
            None => FrozenLakeTerrain::WALL,
            Some(data) => match data.get(col..col + 1) {
                None => FrozenLakeTerrain::WALL,
                Some(tile) => match tile {
                    "S" => FrozenLakeTerrain::START,
                    "F" => FrozenLakeTerrain::GROUND,
                    "G" => FrozenLakeTerrain::GOAL,
                    "H" => FrozenLakeTerrain::HOLE,
                    _ => panic!("Invalid repr on Map!"),
                },
            },
        };
    }

    pub fn new(map: &[&str], is_slippery: bool, max_steps: u128) -> Self {
        let nrow: usize = map.len();
        let ncol: usize = map[0].len();
        let map: Vec<String> = map.iter().map(|s| s.to_string()).collect();
        // calculating start positions probabilities
        let flat_map: String = map.join("");
        let mut initial_state_distrib: Vec<f64> = vec![0.0; flat_map.len()];
        let mut counter = 0;
        let mut pos = vec![];
        for (i, c) in flat_map.char_indices() {
            if c == 'S' {
                counter += 1;
                pos.push(i);
            }
        }
        for i in pos {
            initial_state_distrib[i] = 1.0 / counter as f64;
        }
        // calculating transitions probabilities
        let mut probs: Vec<[[Transition; 3]; 4]> =
            vec![[[(0.0, FrozenLakeObs::default(), 0, 0.0, false); 3]; 4]; nrow * ncol];
        for row in 0..nrow {
            for col in 0..ncol {
                let curr_pos = from_2d_to_1d(ncol, row, col);
                let s = Self::get_obs(&map, row, col);
                for a in 0..4 {
                    let li = &mut probs[curr_pos][a];
                    let letter = map[row].get(col..col + 1).unwrap();
                    if "GH".contains(letter) {
                        li[0] = (1.0, s, curr_pos, 0.0, true)
                    } else if is_slippery {
                        for (i, b) in [(a - 1) % 4, a, (a + 1) % 4].iter().enumerate() {
                            let (s, p, r, t) =
                                Self::update_probability_matrix(&map, nrow, ncol, row, col, *b);
                            li[i] = (1.0 / 3.0, s, p, r, t);
                        }
                    } else {
                        let (s, p, r, t) =
                            Self::update_probability_matrix(&map, nrow, ncol, row, col, a);
                        li[0] = (1.0, s, p, r, t);
                    }
                }
            }
        }

        Self {
            ready: false,
            initial_state_distrib,
            probs,
            player_pos: 0,
            dist: Uniform::from(0.0..1.0),
            max_steps,
            curr_step: 0,
            map,
            ncol,
        }
    }
}

impl Env<FrozenLakeObs, usize> for FrozenLakePartialEnv {
    fn reset(&mut self) -> FrozenLakeObs {
        let dist: Uniform<f64> = Uniform::from(0.0..1.0);
        let random: f64 = dist.sample(&mut rand::thread_rng());
        self.player_pos = categorical_sample(&self.initial_state_distrib.to_vec(), random);
        self.ready = true;
        self.curr_step = 0;
        let (row, col) = from_1d_to_2d(self.ncol, self.player_pos);
        Self::get_obs(&self.map, row, col)
    }

    fn step(&mut self, action: usize) -> Result<(FrozenLakeObs, f64, bool), EnvNotReady> {
        if !self.ready {
            return Err(EnvNotReady);
        }
        if self.curr_step >= self.max_steps {
            self.ready = false;
            let (row, col) = from_1d_to_2d(self.ncol, self.player_pos);
            return Ok((Self::get_obs(&self.map, row, col), -1.0, true));
        }
        self.curr_step += 1;
        let transitions = self.probs[self.player_pos][action];
        let t_probs = transitions.map(|a| a.0);
        let random: f64 = self.dist.sample(&mut rand::thread_rng());
        let i = categorical_sample(t_probs.as_ref(), random);
        let (_p, s, p, r, t) = transitions[i];
        self.player_pos = p;
        if t {
            self.ready = false;
        }
        Ok((s, r, t))
    }

    fn render(&self) -> String {
        let mut new_map: String = self.map.join("\n");
        for (i, _) in new_map.clone().match_indices('S') {
            new_map.replace_range(i..i + 1, "F");
        }
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
