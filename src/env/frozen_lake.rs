use crate::env::{Env, ActionSpace, EnvNotReady};
use crate::utils::{categorical_sample, inc, to_s};

use rand::prelude::Distribution;
use rand::distributions::Uniform;

#[derive(Debug, Clone)]
pub struct FrozenLakeEnv {
    ready: bool,
    initial_state_distrib: Vec<f64>,
    probs: Vec<[[(f64, usize, f64, bool); 3]; 4]>,
    player_pos: usize,
    dist: Uniform<f64>,
    max_steps: u128,
    curr_step: u128
}

impl FrozenLakeEnv {
    // default 4x4 map
    pub const MAP_4X4: [&str;4] = [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ];

    pub const MAP_8X8: [&str;8] = [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ];


    fn update_probability_matrix(map: &[&str], nrow: usize, ncol: usize, row: usize, col: usize, action: usize) -> (usize, f64, bool) {
        let (newrow, newcol) = inc(nrow, ncol, row, col, action);
        let newstate: usize = to_s(ncol, newrow, newcol);
        let newletter: &str = map[newrow].get(newcol..newcol+1).unwrap();
        let terminated: bool = "GH".contains(newletter);
        let reward: f64 = (newletter == "G").into();
        return (newstate, reward, terminated)
    }

    pub fn new(map: &[&str], is_slippery: bool, max_steps: u128) -> Self {
        let ncol: usize = map.len();
        let nrow: usize = map[0].len();
        // calculating start positions probabilities
        let flat_map: String = map.join(&"");
        let mut initial_state_distrib: Vec<f64> = vec![0.0;flat_map.len()];
        let mut counter = 0;
        let mut pos = vec![];
        for (i, c) in flat_map.char_indices() {
            if c == 'S' {
                counter += 1;
                pos.push(i);
            }
        }
        for i in pos {
            initial_state_distrib[i] = 1.0/counter as f64;
        }
        // calculating transitions probabilities
        let mut probs: Vec<[[(f64, usize, f64, bool); 3]; 4]> = vec![[[(0.0, 0, 0.0, false);3];4]; nrow * ncol];
        for row in 0..nrow {
            for col in 0..ncol {
                let s = to_s(ncol, row, col);
                for a in 0..4 {
                    let li = &mut probs[s][a];
                    let letter = map[row].get(col..col+1).unwrap();
                    if "GH".contains(letter) {
                        li[0] = (1.0, s, 0.0, true)
                    } else {
                        if is_slippery{
                            for (i,b) in [(a - 1) % 4, a, (a + 1) % 4].iter().enumerate() {
                                let (s, r, t) = Self::update_probability_matrix(map, nrow, ncol, row, col, *b);
                                li[i] = (1.0 / 3.0, s, r, t);
                            }
                        } else {
                            let (s, r, t) = Self::update_probability_matrix(map, nrow, ncol, row, col, a);
                            li[0] = (1.0, s, r, t);
                        }
                    }
                }
            }
        }

        let env: FrozenLakeEnv = Self {
            ready: false,
            initial_state_distrib,
            probs,
            player_pos: 0,
            dist: Uniform::from(0.0..1.0),
            max_steps,
            curr_step: 0
        };
        return env; 
    }
}

impl Env<usize> for FrozenLakeEnv {
    fn reset(&mut self) -> usize {
        let dist: Uniform<f64> = Uniform::from(0.0..1.0);
        let random: f64 = dist.sample(&mut rand::thread_rng());
        self.player_pos = categorical_sample(&self.initial_state_distrib, random);
        self.ready = true;
        self.curr_step = 0;
        return self.player_pos;
    }

    fn step(&mut self, action: usize) -> Result<(usize, f64, bool), EnvNotReady> {
        if !self.ready {
            return Err(EnvNotReady);
        }
        if self.curr_step >= self.max_steps {
            self.ready = false;
            return Ok((0, -1.0, true))
        }
        self.curr_step += 1;
        let transitions = self.probs[self.player_pos][action];
        let t_probs = transitions.map(|a| a.0);
        let random: f64 = self.dist.sample(&mut rand::thread_rng());
        let i = categorical_sample(&t_probs.to_vec(), random);
        let (_p, s, r, t) = transitions[i];
        self.player_pos = s;
        if t {
            self.ready = false;
        }
        return Ok((s, r, t));
    }

    fn action_space(&self) -> ActionSpace {
        return ActionSpace::new(4);
    }

    
}