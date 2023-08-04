use std::convert::From;
use std::f64::consts::PI;
use std::ops::{Add, Mul};

use rand::distributions::Uniform;
use rand::prelude::Distribution;

use crate::env::EnvNotReady;

use crate::Env;

#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub struct PendulumState {
    pub theta: f64,
    pub theta_angular_velocity: f64,
}

impl PendulumState {
    pub fn new(theta: f64, theta_angular_velocity: f64) -> Self {
        Self {
            theta,
            theta_angular_velocity,
        }
    }
}

impl Add for PendulumState {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            theta: self.theta + other.theta,
            theta_angular_velocity: self.theta_angular_velocity + other.theta_angular_velocity,
        }
    }
}

impl Mul<f64> for PendulumState {
    type Output = Self;
    fn mul(self, other: f64) -> Self {
        Self {
            theta: self.theta * other,
            theta_angular_velocity: self.theta_angular_velocity * other,
        }
    }
}

#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub struct PendulumObservation {
    pub theta_cos: f64,
    pub theta_sin: f64,
    pub theta_angular_velocity: f64,
}

impl PendulumObservation {
    pub fn new(theta_cos: f64, theta_sin: f64, theta_angular_velocity: f64) -> Self {
        Self {
            theta_cos,
            theta_sin,
            theta_angular_velocity,
        }
    }
}

impl From<PendulumState> for PendulumObservation {
    fn from(item: PendulumState) -> Self {
        PendulumObservation {
            theta_cos: item.theta.cos(),
            theta_sin: item.theta.cos(),
            theta_angular_velocity: item.theta_angular_velocity,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PendulumEnv {
    ready: bool,
    max_steps: u128,
    curr_step: u128,
    state: PendulumState,
    theta_dist: Uniform<f64>,
    theta_angular_velocity_dist: Uniform<f64>,
}

impl PendulumEnv {
    pub const ACTIONS: [&str; 1] = ["TORQUE"];
    const GRAVITY: f64 = 9.8;
    const MAX_SPEED: f64 = 8.0;
    const MAX_TORQUE: f64 = 2.0;
    const DT: f64 = 0.05;
    const M: f64 = 1.0;
    const L: f64 = 1.0;

    pub fn new(max_steps: u128) -> Self {
        let mut env: PendulumEnv = Self {
            ready: false,
            curr_step: 0,
            max_steps,
            state: PendulumState::default(),
            theta_dist: Uniform::from(-PI..PI),
            theta_angular_velocity_dist: Uniform::from(-1.0..1.0),
        };
        env.state = env.initialize();
        env
    }

    fn initialize(&self) -> PendulumState {
        PendulumState {
            theta: self.theta_dist.sample(&mut rand::thread_rng()),
            theta_angular_velocity: self
                .theta_angular_velocity_dist
                .sample(&mut rand::thread_rng()),
        }
    }
    fn angle_normalize(x: f64) -> f64 {
        ((x + PI) % (2.0 * PI)) - PI
    }
}

impl Default for PendulumEnv {
    fn default() -> Self {
        Self::new(500)
    }
}

impl Env<PendulumObservation, f64> for PendulumEnv {
    fn reset(&mut self) -> PendulumObservation {
        self.state = self.initialize();
        self.ready = true;
        self.curr_step = 0;
        self.state.into()
    }

    fn step(&mut self, action: f64) -> Result<(PendulumObservation, f64, bool), EnvNotReady> {
        if !self.ready {
            return Err(EnvNotReady);
        }
        if self.curr_step >= self.max_steps {
            self.ready = false;
            return Ok((self.state.into(), -1.0, true));
        }
        self.curr_step += 1;

        let th = self.state.theta;
        let thdot = self.state.theta_angular_velocity;

        let u = action.clamp(-Self::MAX_TORQUE, Self::MAX_TORQUE);
        let an = Self::angle_normalize(th);
        let reward = an * an + 0.1 * thdot * thdot + 0.001 * (u * u);
        let newthdot = thdot
            + (3.0 * Self::GRAVITY / (2.0 * Self::L) * th.sin()
                + 3.0 / (Self::M * Self::L * Self::L) * u)
                * Self::DT;
        let newthdot = newthdot.clamp(-Self::MAX_SPEED, Self::MAX_SPEED);
        let newth = th + newthdot * Self::DT;

        self.state = PendulumState {
            theta: newth,
            theta_angular_velocity: newthdot,
        };
        Ok((self.state.into(), -reward, false))
    }

    fn render(&self) -> String {
        "TODO".to_string()
    }
}
