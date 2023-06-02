use std::rc::Rc;

use crate::{env::{Env, ActionSpace, EnvNotReady}, utils::{inc, to_s}, observation::Observation};

#[derive(Hash, Debug, Clone, PartialEq, Eq, Ord, PartialOrd)]
pub struct CliffWalkingObservation {
    pub pos: usize,
}

impl CliffWalkingObservation {
    fn new (pos: usize) -> Self {
        return Self { pos }
    }
}

impl Observation for CliffWalkingObservation {
    fn base_size(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

#[derive(Debug, Clone)]
pub struct CliffWalkingEnv {
    ready: bool,
    obs: [[(usize, f64, bool); 4]; 48],
    player_pos: usize,
    max_steps: u128,
    curr_step: u128
}

impl CliffWalkingEnv {
    const START_POSITION: usize = 36;
    const CLIFF_POSITIONS: [usize; 10] = [37,38,39,40,41,42,43,44,45,46];
    const GOAL_POSITION: usize = 47;
    pub const ACTIONS: [&str;4] = ["LEFT", "DOWN", "RIGHT", "UP"];
    const MAP: &str = "____________\n____________\n____________\n@!!!!!!!!!!G";

    fn update_probability_matrix(row: usize, col: usize, action: usize) -> (usize, f64, bool) {
        let (newrow, newcol) = inc(4, 12, row, col, action);
        let newstate: usize = to_s(12, newrow, newcol);
        let win: bool = newstate==Self::GOAL_POSITION;
        let lose: bool = Self::CLIFF_POSITIONS.contains(&newstate);
        let reward: f64 = if lose {-100.0} else {-1.0};
        return (newstate, reward, lose || win)
    }

    pub fn new(max_steps: u128) -> Self {
        let ncol: usize = 12;
        let nrow: usize = 4;
        // calculating start positions probabilities
        let mut initial_state_distrib: [f64; 48] = [0.0; 48];
        initial_state_distrib[Self::START_POSITION] = 1.0;
        // calculating transitions probabilities
        let mut obs: [[(usize, f64, bool); 4]; 48] = [[(0, 0.0, false);4]; 48];
        for row in 0..nrow {
            for col in 0..ncol {
                for a in 0..4 {
                    let i = to_s(ncol, row, col);
                    let li = &mut obs[i][a];
                    let (s, r, t) = Self::update_probability_matrix(row, col, a);
                    
                    // println!("i:{:?} = row {:?}, col {:?}, action {:?}", i, row, col, a);
                    li.0 = s;
                    li.1 = r;
                    li.2 = t;
                    // println!("{:?} row {:?} col {:?}", li, row, col);
                }
            }
        }

        let env: CliffWalkingEnv = Self {
            ready: false,
            obs,
            player_pos: 0,
            max_steps,
            curr_step: 0
        };
        return env; 
    }
}

impl Env for CliffWalkingEnv {
    fn reset(&mut self) -> Rc<dyn Observation> {
        self.player_pos = Self::START_POSITION;
        self.ready = true;
        self.curr_step = 0;
        return Rc::new(CliffWalkingObservation::new(self.player_pos));
    }

    fn step(&mut self, action: usize) -> Result<(Rc<dyn Observation>, f64, bool), EnvNotReady> {
        if !self.ready {
            return Err(EnvNotReady);
        }
        if self.curr_step >= self.max_steps {
            self.ready = false;
            return Ok((Rc::new(CliffWalkingObservation::new(0)), -100.0, true))
        }
        self.curr_step += 1;
        let (pos, r, t): (usize, f64, bool) = self.obs[self.player_pos][action];
        self.player_pos = pos;
        if t {
            self.ready = false;
        }
        return Ok((Rc::new(CliffWalkingObservation::new(pos)), r, t));
    }

    fn action_space(&self) -> ActionSpace {
        return ActionSpace::new(4);
    }

    fn render(&self) -> String {
        let mut new_map: String = Self::MAP.clone().to_string();
        new_map.replace_range(39..40,"_");
        let mut pos: usize = self.player_pos;
        for (i, _) in new_map.match_indices('\n') {
            if pos >= i {
                pos += 1;
            }
        }
        new_map.replace_range(pos..pos+1,"@");
        return new_map;
    }

    fn get_action_label(&self, action: usize) -> &str {
        return Self::ACTIONS[action]
    }
    
}