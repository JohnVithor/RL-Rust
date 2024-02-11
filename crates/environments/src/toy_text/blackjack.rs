use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::ops::{Index, IndexMut};

use crate::env::{DiscreteAction, DiscreteEnv};
use crate::EnvError;

extern crate core;

use num_enum::IntoPrimitive;
use rand::distributions::Uniform;
use rand::prelude::Distribution;
use rand::rngs::ThreadRng;

#[derive(Debug, Copy, Clone, IntoPrimitive)]
#[repr(usize)]
pub enum BlackJackAction {
    HIT,
    STICK,
}

impl From<usize> for BlackJackAction {
    fn from(value: usize) -> Self {
        match value {
            0 => Self::HIT,
            1 => Self::STICK,
            value => panic!(
                "Invalid value to convert from usize to BlackJackAction: {}",
                value
            ),
        }
    }
}

impl DiscreteAction for BlackJackAction {}

impl Index<BlackJackAction> for [f64] {
    type Output = f64;

    fn index(&self, index: BlackJackAction) -> &Self::Output {
        &self[match index {
            BlackJackAction::HIT => 0,
            BlackJackAction::STICK => 1,
        }]
    }
}

impl IndexMut<BlackJackAction> for [f64] {
    fn index_mut(&mut self, index: BlackJackAction) -> &mut Self::Output {
        &mut self[match index {
            BlackJackAction::HIT => 0,
            BlackJackAction::STICK => 1,
        }]
    }
}

impl Index<BlackJackAction> for [u128] {
    type Output = u128;

    fn index(&self, index: BlackJackAction) -> &Self::Output {
        &self[match index {
            BlackJackAction::HIT => 0,
            BlackJackAction::STICK => 1,
        }]
    }
}

impl IndexMut<BlackJackAction> for [u128] {
    fn index_mut(&mut self, index: BlackJackAction) -> &mut Self::Output {
        &mut self[match index {
            BlackJackAction::HIT => 0,
            BlackJackAction::STICK => 1,
        }]
    }
}

#[derive(Hash, Debug, Clone, PartialEq, Eq, Ord, PartialOrd)]
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
    pub fn get_id(&self) -> usize {
        let mut s = DefaultHasher::new();
        self.hash(&mut s);
        s.finish() as usize
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
        self.dealer[0]
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

impl DiscreteEnv<BlackJackObservation, BlackJackAction> for BlackJackEnv {
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

    //fn render_episode(&self) {}
}
