use std::cmp::Ordering;

use crate::env::DiscreteActionEnv;
use crate::space::{SpaceInfo, SpaceTypeBounds};

use ndarray::{Array1, ArrayD};
use rand::distributions::Uniform;
use rand::prelude::Distribution;
use rand::rngs::SmallRng;
use rand::SeedableRng;

pub enum BlackJackError {
    NotReady,
    InvalidAction,
}

#[derive(Debug, Clone)]
pub struct BlackJackEnv {
    ready: bool,
    player: [u8; 16],
    player_i: usize,
    pub dealer: [u8; 16],
    dealer_i: usize,
    player_has_ace: bool,
    dealer_has_ace: bool,
    dist: Uniform<u8>,
    rng: SmallRng,
}

impl BlackJackEnv {
    pub fn new(seed: u64) -> Self {
        let mut env: BlackJackEnv = Self {
            ready: false,
            player: [0; 16],
            player_i: 0,
            dealer: [0; 16],
            dealer_i: 0,
            player_has_ace: false,
            dealer_has_ace: false,
            rng: SmallRng::seed_from_u64(seed),
            dist: Uniform::from(1..11),
        };
        env.initialize_hands();
        env
    }
    fn initialize_hands(&mut self) {
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
        if self.dealer[0] == 1 {
            11
        } else {
            self.dealer[0]
        }
    }

    fn get_new_card(&mut self) -> u8 {
        self.dist.sample(&mut self.rng)
    }

    fn compute_player_score(&self) -> u8 {
        let score: u8 = self.player.iter().sum();
        if self.player_has_ace && score + 10 <= 21 {
            score + 10
        } else {
            score
        }
    }

    fn compute_dealer_score(&self) -> u8 {
        let score: u8 = self.dealer.iter().sum();
        if self.dealer_has_ace && score + 10 <= 21 {
            score + 10
        } else {
            score
        }
    }

    fn build_obs(&self, p_score: u8, d_score: u8, has_ace: bool) -> ArrayD<f32> {
        Array1::from_iter([p_score as f32, d_score as f32, has_ace as u8 as f32]).into_dyn()
    }
}

impl Default for BlackJackEnv {
    fn default() -> Self {
        Self::new(42)
    }
}

impl DiscreteActionEnv for BlackJackEnv {
    type Error = BlackJackError;
    fn reset(&mut self) -> Result<ArrayD<f32>, BlackJackError> {
        self.player = [0; 16];
        self.dealer = [0; 16];
        self.initialize_hands();
        self.ready = true;
        Ok(self.build_obs(
            self.compute_player_score(),
            self.get_dealer_card(),
            self.player_has_ace,
        ))
    }

    fn observation_space(&self) -> SpaceInfo {
        SpaceInfo::new(vec![
            SpaceTypeBounds::Discrete(26),
            SpaceTypeBounds::Discrete(28),
            SpaceTypeBounds::Discrete(2),
        ])
    }
    fn action_space(&self) -> SpaceInfo {
        SpaceInfo::new(vec![SpaceTypeBounds::Discrete(2)])
    }

    fn step(&mut self, action: usize) -> Result<(ArrayD<f32>, f32, bool), BlackJackError> {
        if !self.ready {
            return Err(BlackJackError::NotReady);
        }
        match action {
            0 => {
                self.player[self.player_i] = self.get_new_card();
                self.player_i += 1;
                let p_score: u8 = self.compute_player_score();
                if p_score > 21 {
                    self.ready = false;
                    let obs =
                        self.build_obs(p_score, self.compute_dealer_score(), self.player_has_ace);
                    return Ok((obs, -1.0, true));
                }
                let obs = self.build_obs(p_score, self.get_dealer_card(), self.player_has_ace);
                Ok((obs, 0.0, false))
            }
            1 => {
                self.ready = false;
                let mut d_score: u8 = self.compute_dealer_score();
                while d_score < 17 {
                    self.dealer[self.dealer_i] = self.get_new_card();
                    self.dealer_i += 1;
                    d_score = self.compute_dealer_score();
                }
                let p_score = self.compute_player_score();
                let obs = self.build_obs(p_score, d_score, self.player_has_ace);
                if d_score > 21 {
                    return Ok((obs, 1.0, true));
                }

                let reward = match p_score.cmp(&d_score) {
                    Ordering::Greater => 1.0,
                    Ordering::Less => -1.0,
                    Ordering::Equal => 0.0,
                };

                Ok((obs, reward, true))
            }
            _ => Err(BlackJackError::InvalidAction),
        }
    }

    fn render(&self) -> String {
        let mut result;
        if self.ready {
            result = format!("Dealer: {} \nPlayer: ", self.dealer[0]);
        } else {
            let mut dealer_cards = "".to_string();
            for i in &self.dealer[0..self.dealer_i] {
                dealer_cards.push_str(i.to_string().as_str());
                dealer_cards.push(' ');
            }
            result = format!("Dealer: {} \nPlayer: ", dealer_cards);
        }
        let mut player_cards = "".to_string();
        for i in &self.player[0..self.player_i] {
            player_cards.push_str(i.to_string().as_str());
            player_cards.push(' ');
        }
        result.push_str(&player_cards);
        result
    }
}
