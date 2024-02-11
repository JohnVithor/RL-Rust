use std::cmp::Ordering;

use crate::env::Env;
use crate::space::{SpaceInfo, SpaceTypeBounds};
use crate::EnvError;

use rand::distributions::Uniform;
use rand::prelude::Distribution;
use rand::rngs::ThreadRng;

#[derive(Debug, Copy, Clone)]
pub enum BlackJackAction {
    HIT,
    STICK,
}

#[derive(Hash, PartialEq, Eq, Debug, Clone, Copy)]
pub struct BlackJackObservation {
    pub p_score: u8,
    pub d_score: u8,
    pub p_ace: bool,
}

impl BlackJackObservation {
    pub fn new(p_score: u8, d_score: u8, p_ace: bool) -> Self {
        Self {
            p_score,
            d_score,
            p_ace,
        }
    }
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
    rng: ThreadRng,
    dist: Uniform<u8>,
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
}

impl Default for BlackJackEnv {
    fn default() -> Self {
        Self::new()
    }
}

impl Env<BlackJackObservation, BlackJackAction> for BlackJackEnv {
    fn reset(&mut self) -> BlackJackObservation {
        self.player = [0; 16];
        self.dealer = [0; 16];
        self.initialize_hands();
        let obs: BlackJackObservation = BlackJackObservation::new(
            self.compute_player_score(),
            self.get_dealer_card(),
            self.player_has_ace,
        );
        self.ready = true;
        obs
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

    fn step(
        &mut self,
        action: BlackJackAction,
    ) -> Result<(BlackJackObservation, f64, bool), EnvError> {
        if !self.ready {
            return Err(EnvError::EnvNotReady);
        }
        match action {
            BlackJackAction::HIT => {
                self.player[self.player_i] = self.get_new_card();
                self.player_i += 1;
                let p_score: u8 = self.compute_player_score();
                if p_score > 21 {
                    self.ready = false;
                    let obs: BlackJackObservation = BlackJackObservation::new(
                        p_score,
                        self.compute_dealer_score(),
                        self.player_has_ace,
                    );
                    return Ok((obs, -1.0, true));
                }
                let obs: BlackJackObservation =
                    BlackJackObservation::new(p_score, self.get_dealer_card(), self.player_has_ace);
                Ok((obs, 0.0, false))
            }
            BlackJackAction::STICK => {
                self.ready = false;
                let mut d_score: u8 = self.compute_dealer_score();
                while d_score < 17 {
                    self.dealer[self.dealer_i] = self.get_new_card();
                    self.dealer_i += 1;
                    d_score = self.compute_dealer_score();
                }
                let obs: BlackJackObservation = BlackJackObservation::new(
                    self.compute_player_score(),
                    d_score,
                    self.player_has_ace,
                );
                if d_score > 21 {
                    return Ok((obs, 1.0, true));
                }

                let reward = match obs.p_score.cmp(&d_score) {
                    Ordering::Greater => 1.0,
                    Ordering::Less => -1.0,
                    Ordering::Equal => 0.0,
                };

                Ok((obs, reward, true))
            }
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
