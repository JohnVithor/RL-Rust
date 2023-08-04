use std::convert::From;
use std::f32::consts::PI;
use std::ops::{Add, Mul};

use rand::distributions::Uniform;
use rand::prelude::Distribution;

use crate::env::EnvNotReady;

use crate::utils::{bound, wrap};
use crate::Env;

#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub struct AcrobotState {
    pub theta1: f32,
    pub theta2: f32,
    pub theta1_angular_velocity: f32,
    pub theta2_angular_velocity: f32,
}

impl AcrobotState {
    pub fn new(
        theta1: f32,
        theta2: f32,
        theta1_angular_velocity: f32,
        theta2_angular_velocity: f32,
    ) -> Self {
        Self {
            theta1,
            theta2,
            theta1_angular_velocity,
            theta2_angular_velocity,
        }
    }
}

impl Add for AcrobotState {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            theta1: self.theta1 + other.theta1,
            theta2: self.theta2 + other.theta2,
            theta1_angular_velocity: self.theta1_angular_velocity + other.theta1_angular_velocity,
            theta2_angular_velocity: self.theta2_angular_velocity + other.theta2_angular_velocity,
        }
    }
}

impl Mul<f32> for AcrobotState {
    type Output = Self;
    fn mul(self, other: f32) -> Self {
        Self {
            theta1: self.theta1 * other,
            theta2: self.theta2 * other,
            theta1_angular_velocity: self.theta1_angular_velocity * other,
            theta2_angular_velocity: self.theta2_angular_velocity * other,
        }
    }
}

#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub struct AcrobotObservation {
    pub theta1_cos: f32,
    pub theta1_sin: f32,
    pub theta2_cos: f32,
    pub theta2_sin: f32,
    pub theta1_angular_velocity: f32,
    pub theta2_angular_velocity: f32,
}

impl AcrobotObservation {
    pub fn new(
        theta1_cos: f32,
        theta1_sin: f32,
        theta2_cos: f32,
        theta2_sin: f32,
        theta1_angular_velocity: f32,
        theta2_angular_velocity: f32,
    ) -> Self {
        Self {
            theta1_cos,
            theta1_sin,
            theta2_cos,
            theta2_sin,
            theta1_angular_velocity,
            theta2_angular_velocity,
        }
    }
}

impl From<AcrobotState> for AcrobotObservation {
    fn from(item: AcrobotState) -> Self {
        AcrobotObservation {
            theta1_cos: item.theta1.cos(),
            theta1_sin: item.theta1.cos(),
            theta2_cos: item.theta2.sin(),
            theta2_sin: item.theta2.sin(),
            theta1_angular_velocity: item.theta1_angular_velocity,
            theta2_angular_velocity: item.theta2_angular_velocity,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AcrobotEnv {
    ready: bool,
    max_steps: u128,
    curr_step: u128,
    state: AcrobotState,
    dist: Uniform<f32>,
    torque_dist: Uniform<f32>,
    torque_noise_max: f32,
}

impl AcrobotEnv {
    pub const ACTIONS: [&str; 3] = [
        "APPLY TORQUE TO THE LEFT",
        "DONT APPLY TORQUE",
        "APPLY TORQUE TO THE RIGTH",
    ];
    const GRAVITY: f32 = 9.8;
    const DT: f32 = 0.2;
    const LINK_LENGTH_1: f32 = 1.0;
    const _LINK_LENGTH_2: f32 = 1.0;
    const LINK_MASS_1: f32 = 1.0;
    const LINK_MASS_2: f32 = 1.0;
    const LINK_COM_POS_1: f32 = 0.5;
    const LINK_COM_POS_2: f32 = 0.5;
    const LINK_MOI: f32 = 1.0;
    const MAX_VEL_1: f32 = 4.0 * PI;
    const MAX_VEL_2: f32 = 9.0 * PI;
    const AVAIL_TORQUE: [f32; 3] = [-1.0, 0.0, 1.0];

    pub fn new(max_steps: u128, torque_noise_max: f32) -> Self {
        let mut env: AcrobotEnv = Self {
            ready: false,
            curr_step: 0,
            max_steps,
            state: AcrobotState::default(),
            dist: Uniform::from(-0.1..0.1),
            torque_dist: Uniform::from(-torque_noise_max..torque_noise_max),
            torque_noise_max,
        };
        env.state = env.initialize();
        env
    }

    fn initialize(&self) -> AcrobotState {
        AcrobotState {
            theta1: self.dist.sample(&mut rand::thread_rng()),
            theta2: self.dist.sample(&mut rand::thread_rng()),
            theta1_angular_velocity: self.dist.sample(&mut rand::thread_rng()),
            theta2_angular_velocity: self.dist.sample(&mut rand::thread_rng()),
        }
    }

    fn dsdt(state: AcrobotState, torque: f32) -> AcrobotState {
        let d1 = Self::LINK_MASS_1 * Self::LINK_COM_POS_1 * Self::LINK_COM_POS_1
            + Self::LINK_MASS_2
                * (Self::LINK_LENGTH_1 * Self::LINK_LENGTH_1
                    + Self::LINK_COM_POS_2 * Self::LINK_COM_POS_2
                    + 2.0 * Self::LINK_LENGTH_1 * Self::LINK_COM_POS_2 * state.theta2.cos())
            + Self::LINK_MOI
            + Self::LINK_MOI;
        let d2 = Self::LINK_MASS_2
            * (Self::LINK_COM_POS_2 * Self::LINK_COM_POS_2
                + Self::LINK_LENGTH_1 * Self::LINK_COM_POS_2 * state.theta2.cos())
            + Self::LINK_MOI;
        let phi2 = Self::LINK_MASS_2
            * Self::LINK_COM_POS_2
            * Self::GRAVITY
            * (state.theta1 + state.theta2 - PI / 2.0).cos();
        let phi1 = -Self::LINK_MASS_2
            * Self::LINK_LENGTH_1
            * Self::LINK_COM_POS_2
            * state.theta2_angular_velocity
            * state.theta2_angular_velocity
            * state.theta2.sin()
            - 2.0
                * Self::LINK_MASS_2
                * Self::LINK_LENGTH_1
                * Self::LINK_COM_POS_2
                * state.theta2_angular_velocity
                * state.theta1_angular_velocity
                * state.theta2.sin()
            + (Self::LINK_MASS_1 * Self::LINK_COM_POS_1 + Self::LINK_MASS_2 * Self::LINK_LENGTH_1)
                * Self::GRAVITY
                * (state.theta1 - PI / 2.0).cos()
            + phi2;

        let ddtheta2 = (torque + d2 / d1 * phi1
            - Self::LINK_MASS_2
                * Self::LINK_LENGTH_1
                * Self::LINK_COM_POS_2
                * state.theta1_angular_velocity
                * state.theta1_angular_velocity
                * state.theta2.sin()
            - phi2)
            / (Self::LINK_MASS_2 * Self::LINK_COM_POS_2 * Self::LINK_COM_POS_2 + Self::LINK_MOI
                - d2 * d2 / d1);

        let ddtheta1 = -(d2 * ddtheta2 + phi1) / d1;
        AcrobotState {
            theta1: state.theta1_angular_velocity,
            theta2: state.theta2_angular_velocity,
            theta1_angular_velocity: ddtheta1,
            theta2_angular_velocity: ddtheta2,
        }
    }
}

impl Default for AcrobotEnv {
    fn default() -> Self {
        Self::new(500, 0.0)
    }
}

impl Env<AcrobotObservation, usize> for AcrobotEnv {
    fn reset(&mut self) -> AcrobotObservation {
        self.state = self.initialize();
        self.ready = true;
        self.curr_step = 0;
        self.state.into()
    }

    fn step(&mut self, action: usize) -> Result<(AcrobotObservation, f64, bool), EnvNotReady> {
        if !self.ready {
            return Err(EnvNotReady);
        }
        if self.curr_step >= self.max_steps {
            self.ready = false;
            return Ok((self.state.into(), -1.0, true));
        }
        self.curr_step += 1;

        let mut torque = Self::AVAIL_TORQUE[action];
        if self.torque_noise_max > 0.0 {
            torque += self.torque_dist.sample(&mut rand::thread_rng());
        }

        let initial_state = self.state;
        let k1 = Self::dsdt(initial_state, torque);
        let k2 = Self::dsdt(initial_state + k1 * (Self::DT / 2.0), torque);
        let k3 = Self::dsdt(initial_state + k2 * (Self::DT / 2.0), torque);
        let k4 = Self::dsdt(initial_state + k3 * Self::DT, torque);
        let new_state = initial_state + (k1 + (k2 * 2.0) + (k3 * 2.0) + k4) * (Self::DT / 6.0);

        self.state = AcrobotState {
            theta1: wrap(new_state.theta1, -PI, PI),
            theta2: wrap(new_state.theta2, -PI, PI),
            theta1_angular_velocity: bound(
                new_state.theta1_angular_velocity,
                -Self::MAX_VEL_1,
                Self::MAX_VEL_1,
            ),
            theta2_angular_velocity: bound(
                new_state.theta2_angular_velocity,
                -Self::MAX_VEL_2,
                Self::MAX_VEL_2,
            ),
        };
        let terminated =
            -self.state.theta1.cos() - (self.state.theta2 + self.state.theta1).cos() > 1.0;
        let reward = if !terminated { -1.0 } else { 0.0 };
        Ok((self.state.into(), reward, terminated))
    }

    fn render(&self) -> String {
        "TODO".to_string()
    }
}
