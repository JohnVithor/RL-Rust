use rand::distributions::Uniform;
use rand::prelude::Distribution;

use crate::space::{SpaceInfo, SpaceTypeBounds};
use crate::EnvError::EnvNotReady;
use crate::{Env, EnvError};

#[derive(Debug, Clone, Default, PartialEq)]
pub struct CartPoleObservation {
    pub cart_position: f32,
    pub cart_velocity: f32,
    pub pole_angle: f32,
    pub pole_angular_velocity: f32,
}

impl CartPoleObservation {
    pub fn new(
        cart_position: f32,
        cart_velocity: f32,
        pole_angle: f32,
        pole_angular_velocity: f32,
    ) -> Self {
        Self {
            cart_position,
            cart_velocity,
            pole_angle,
            pole_angular_velocity,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CartPoleEnv {
    ready: bool,
    max_steps: u128,
    curr_step: u128,
    state: CartPoleObservation,
    dist: Uniform<f32>,
}

impl CartPoleEnv {
    pub const ACTIONS: [&'static str; 2] = ["PUSH TO THE LEFT", "PUSH TO THE RIGTH"];
    const GRAVITY: f32 = 9.8;
    const POLE_MASS: f32 = 0.1;
    const TOTAL_MASS: f32 = 1.1;
    const POLE_HALF_LENGTH: f32 = 0.5;
    const POLE_MASS_LENGTH: f32 = 0.05;
    const FORCE_MAG: f32 = 10.0;
    const TAU: f32 = 0.02;
    const THETA_THRESHOLD_RADIANS: f32 = std::f32::consts::PI / 15.0;
    const X_THRESHOLD: f32 = 2.4;

    pub fn new(max_steps: u128) -> Self {
        let mut env: CartPoleEnv = Self {
            ready: false,
            curr_step: 0,
            max_steps,
            state: CartPoleObservation::default(),
            dist: Uniform::from(-0.05..0.05),
        };
        env.state = env.initialize();
        env
    }

    fn initialize(&self) -> CartPoleObservation {
        CartPoleObservation {
            cart_position: self.dist.sample(&mut rand::thread_rng()),
            cart_velocity: self.dist.sample(&mut rand::thread_rng()),
            pole_angle: self.dist.sample(&mut rand::thread_rng()),
            pole_angular_velocity: self.dist.sample(&mut rand::thread_rng()),
        }
    }
}

impl Default for CartPoleEnv {
    fn default() -> Self {
        Self::new(500)
    }
}

impl Env<CartPoleObservation, usize> for CartPoleEnv {
    fn reset(&mut self) -> CartPoleObservation {
        self.state = self.initialize();
        self.ready = true;
        self.curr_step = 0;
        self.state.clone()
    }

    fn step(&mut self, action: usize) -> Result<(CartPoleObservation, f64, bool), EnvError> {
        if !self.ready {
            return Err(EnvNotReady);
        }
        if self.curr_step >= self.max_steps {
            self.ready = false;
            return Ok((self.state.clone(), -1.0, true));
        }
        self.curr_step += 1;

        let force = if action == 1 {
            Self::FORCE_MAG
        } else {
            -Self::FORCE_MAG
        };
        let cos_theta = self.state.pole_angle.cos();
        let sin_theta = self.state.pole_angle.sin();

        let temp = (force
            + Self::POLE_MASS_LENGTH
                * self.state.pole_angular_velocity
                * self.state.pole_angular_velocity
                * sin_theta)
            / Self::TOTAL_MASS;
        let thetaacc = (Self::GRAVITY * sin_theta - cos_theta * temp)
            / (Self::POLE_HALF_LENGTH
                * (4.0 / 3.0 - Self::POLE_MASS * cos_theta * cos_theta / Self::TOTAL_MASS));
        let xacc = temp - Self::POLE_MASS_LENGTH * thetaacc * cos_theta / Self::TOTAL_MASS;

        self.state.cart_position += Self::TAU * self.state.cart_velocity;
        self.state.cart_velocity += Self::TAU * xacc;
        self.state.pole_angle += Self::TAU * self.state.pole_angular_velocity;
        self.state.pole_angular_velocity += Self::TAU * thetaacc;

        let terminated = self.state.cart_position < -Self::X_THRESHOLD
            || self.state.cart_position > Self::X_THRESHOLD
            || self.state.pole_angle < -Self::THETA_THRESHOLD_RADIANS
            || self.state.pole_angle > Self::THETA_THRESHOLD_RADIANS;
        let reward = if !terminated { 1.0 } else { 0.0 };
        Ok((self.state.clone(), reward, terminated))
    }

    fn render(&self) -> String {
        "TODO".to_string()
    }

    fn observation_space(&self) -> SpaceInfo {
        SpaceInfo::new(vec![
            SpaceTypeBounds::Continuous(-4.8, 4.8),
            SpaceTypeBounds::Continuous(f64::NEG_INFINITY, f64::INFINITY),
            SpaceTypeBounds::Continuous(-0.418, 0.418),
            SpaceTypeBounds::Continuous(f64::NEG_INFINITY, f64::INFINITY),
        ])
    }
    fn action_space(&self) -> SpaceInfo {
        SpaceInfo::new(vec![SpaceTypeBounds::Discrete(2)])
    }
}
