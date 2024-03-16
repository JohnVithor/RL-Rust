use std::ops::Index;

use crate::env::{Env, EnvError};
use crate::space::{SpaceInfo, SpaceTypeBounds};
use crate::utils::{from_2d_to_1d, inc};

use fastrand::Rng;
use utils::categorical_sample;

#[derive(Debug, Copy, Clone)]
pub enum FrozenLakeAction {
    LEFT,
    DOWN,
    RIGHT,
    UP,
}

impl Index<FrozenLakeAction> for [(usize, f32, bool); 4] {
    type Output = (usize, f32, bool);

    fn index(&self, index: FrozenLakeAction) -> &Self::Output {
        match index {
            FrozenLakeAction::LEFT => &self[0],
            FrozenLakeAction::DOWN => &self[1],
            FrozenLakeAction::RIGHT => &self[2],
            FrozenLakeAction::UP => &self[3],
        }
    }
}

type Transition = (f32, usize, f32, bool);

#[derive(Debug, Clone)]
pub struct FrozenLakeEnv {
    ready: bool,
    initial_state_distrib: Vec<f32>,
    probs: Vec<[[Transition; 3]; 4]>,
    player_pos: usize,
    max_steps: u128,
    curr_step: u128,
    map: String,
    rng: Rng,
}

impl FrozenLakeEnv {
    // default 4x4 map
    pub const MAP_4X4: [&'static str; 4] = ["SFFF", "FHFH", "FFFH", "HFFG"];

    pub const MAP_8X8: [&'static str; 8] = [
        "SFFFFFFF", "FFFFFFFF", "FFFHFFFF", "FFFFFHFF", "FFFHFFFF", "FHHFFFHF", "FHFFHFHF",
        "FFFHFFFG",
    ];

    pub const ACTIONS: [&'static str; 4] = ["LEFT", "DOWN", "RIGHT", "UP"];

    fn update_probability_matrix(
        map: &[&str],
        nrow: usize,
        ncol: usize,
        row: usize,
        col: usize,
        action: usize,
    ) -> (usize, f32, bool) {
        let (newrow, newcol) = inc(nrow, ncol, row, col, action);
        let newstate: usize = from_2d_to_1d(ncol, newrow, newcol);
        let newletter: &str = map[newrow].get(newcol..newcol + 1).unwrap();
        let terminated: bool = "GH".contains(newletter);
        let reward: f32 = (newletter == "G").into();
        (newstate, reward, terminated)
    }

    pub fn new(map: &[&str], is_slippery: bool, max_steps: u128, seed: u64) -> Self {
        let nrow: usize = map.len();
        let ncol: usize = map[0].len();
        // calculating start positions probabilities
        let flat_map: String = map.join("");
        let mut initial_state_distrib: Vec<f32> = vec![0.0; flat_map.len()];
        let mut counter = 0;
        let mut pos = vec![];
        for (i, c) in flat_map.char_indices() {
            if c == 'S' {
                counter += 1;
                pos.push(i);
            }
        }
        for i in pos {
            initial_state_distrib[i] = 1.0 / counter as f32;
        }
        // calculating transitions probabilities
        let mut probs: Vec<[[Transition; 3]; 4]> =
            vec![[[(0.0, 0, 0.0, false); 3]; 4]; nrow * ncol];
        for row in 0..nrow {
            for col in 0..ncol {
                let s = from_2d_to_1d(ncol, row, col);
                for a in 0..4 {
                    let li = &mut probs[s][a];
                    let letter = map[row].get(col..col + 1).unwrap();
                    if "GH".contains(letter) {
                        li[0] = (1.0, s, 0.0, true)
                    } else if is_slippery {
                        for (i, b) in [(a - 1) % 4, a, (a + 1) % 4].iter().enumerate() {
                            let (s, r, t) =
                                Self::update_probability_matrix(map, nrow, ncol, row, col, *b);
                            li[i] = (1.0 / 3.0, s, r, t);
                        }
                    } else {
                        let (s, r, t) =
                            Self::update_probability_matrix(map, nrow, ncol, row, col, a);
                        li[0] = (1.0, s, r, t);
                    }
                }
            }
        }

        Self {
            ready: false,
            initial_state_distrib,
            probs,
            player_pos: 0,
            max_steps,
            curr_step: 0,
            map: map.join("\n"),
            rng: Rng::with_seed(seed),
        }
    }
}

impl Env<usize, FrozenLakeAction> for FrozenLakeEnv {
    fn reset(&mut self) -> usize {
        let random: f32 = self.rng.f32();
        self.player_pos = categorical_sample(&self.initial_state_distrib, random);
        self.ready = true;
        self.curr_step = 0;
        self.player_pos
    }

    fn observation_space(&self) -> SpaceInfo {
        SpaceInfo::new(vec![SpaceTypeBounds::Discrete(self.probs.len())])
    }

    fn action_space(&self) -> SpaceInfo {
        SpaceInfo::new(vec![SpaceTypeBounds::Discrete(4)])
    }

    fn step(&mut self, action: FrozenLakeAction) -> Result<(usize, f32, bool), EnvError> {
        if !self.ready {
            return Err(EnvError::EnvNotReady);
        }
        if self.curr_step >= self.max_steps {
            self.ready = false;
            return Ok((0, 0.0, true));
        }
        self.curr_step += 1;
        let transitions = self.probs[self.player_pos][action as usize];
        let t_probs = transitions.map(|a| a.0);
        let random: f32 = self.rng.f32();
        let i = categorical_sample(t_probs.as_ref(), random);
        let (_p, s, r, t) = transitions[i];
        self.player_pos = s;
        if t {
            self.ready = false;
        }
        Ok((s, r, t))
    }

    fn render(&self) -> String {
        let mut new_map: String = self.map.clone();
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
