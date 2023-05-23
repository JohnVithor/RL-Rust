use crate::env::{Env, Observation, ActionSpace, EnvNotReady};

use rand::prelude::Distribution;
use rand::rngs::ThreadRng;
use rand::distributions::Uniform;

#[derive(Debug, Clone)]
pub struct BlackJackEnv {
    ready: bool,
    player: [u8; 16],
    player_i: usize,
    pub dealer: [u8; 16],
    dealer_i: usize,
    player_has_ace: bool,
    dealer_has_ace: bool,
    rng: ThreadRng,
    dist: Uniform<u8>
}

impl BlackJackEnv {
    pub fn new() -> Self {
        let mut env: BlackJackEnv = Self {
            ready: false,
            player: [0; 16],
            player_i: 0,
            dealer: [0; 16],
            dealer_i: 0,
            player_has_ace: false,
            dealer_has_ace: false,
            rng: rand::thread_rng(),
            dist: Uniform::from(1..11)
        };
        env.initialize_hands();
        return env; 
    }
    fn initialize_hands(&mut self){
        self.player[0] = self.get_new_card();
        self.player[1] = self.get_new_card();
        self.player_i = 2;
        self.dealer[0] = self.get_new_card();
        self.dealer[1] = self.get_new_card();
        self.dealer_i = 2;
        self.player_has_ace = (self.player[0] == 1) || (self.player[1] == 1);
        self.dealer_has_ace = (self.dealer[0] == 1) || (self.dealer[1] == 1)
    }

    fn get_dealer_card(&self) -> u8 {
        return self.dealer[0];
    }

    fn get_new_card(&mut self) -> u8 {
        return self.dist.sample(&mut self.rng);
    }

    fn compute_player_score(&self) -> u8{
        let score: u8 = self.player.iter().sum();
        if self.player_has_ace && score+10 <= 21 {
            return score+10
        } else {
            return score
        }
    }

    fn compute_dealer_score(&self) -> u8{
        let score: u8 = self.dealer.iter().sum();
        if self.dealer_has_ace && score+10 <= 21 {
            return score+10
        } else {
            return score
        }
    }
}

impl Env for BlackJackEnv {
    fn reset(&mut self) -> Observation {
        self.player = [0;16];
        self.dealer = [0;16];
        self.initialize_hands();
        let obs: Observation = Observation::new(self.compute_player_score(), self.get_dealer_card(), self.player_has_ace);
        self.ready = true;
        return obs
    }

    fn step(&mut self, action: usize) -> Result<(Observation, f64, bool), EnvNotReady> {
        if !self.ready {
            return Err(EnvNotReady);
        }
        if action == 0{
            self.player[self.player_i] = self.get_new_card();
            self.player_i += 1;
            let p_score: u8 = self.compute_player_score();
            if p_score > 21 {
                self.ready = false;
                let obs: Observation = Observation::new(p_score, self.compute_dealer_score(), self.player_has_ace);
                return Ok((obs, -1.0, true));
            }
            let obs: Observation = Observation::new(p_score, self.get_dealer_card(), self.player_has_ace);
            return Ok((obs, 0.0, false));
        } else {
            self.ready = false;
            let mut d_score: u8 = self.compute_dealer_score();
            while d_score < 17 {
                self.dealer[self.dealer_i] = self.get_new_card();
                self.dealer_i += 1;
                d_score = self.compute_dealer_score();
            }
            let obs: Observation = Observation::new(self.compute_player_score(), d_score, self.player_has_ace);
            if d_score > 21{
                return Ok((obs, 1.0, true));
            }
            let reward: f64 = if obs.p_score > d_score {1.0} else if d_score > obs.p_score {-1.0} else {0.0};
            return Ok((obs, reward, true));
        }
    }

    fn action_space(&self) -> ActionSpace {
        return ActionSpace::new(2);
    }

    
}