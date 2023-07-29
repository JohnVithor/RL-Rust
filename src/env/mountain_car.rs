use rand::distributions::Uniform;
use rand::prelude::Distribution;

use crate::env::EnvNotReady;

use super::Env;

#[derive(Debug, Clone, Default)]
pub struct MountainCarObservation {
    pub position: f32,
    pub velocity: f32,
}

impl MountainCarObservation {
    pub fn new(position: f32, velocity: f32) -> Self {
        Self { position, velocity }
    }
}

#[derive(Debug, Clone)]
pub struct MountainCarEnv {
    ready: bool,
    max_steps: u128,
    curr_step: u128,

    min_position: f32,
    max_position: f32,
    max_speed: f32,
    goal_position: f32,
    goal_velocity: f32,
    force: f32,
    gravity: f32,
    state: MountainCarObservation,

    dist: Uniform<f32>,
}

impl MountainCarEnv {
    pub const ACTIONS: [&str; 3] = [
        "ACCELERATE TO THE LEFT",
        "DONT ACCELERATE",
        "ACCELERATE TO THE RIGTH",
    ];

    pub fn new(max_steps: u128) -> Self {
        let mut env: MountainCarEnv = Self {
            ready: false,
            curr_step: 0,
            max_steps,
            min_position: -1.2,
            max_position: 0.6,
            max_speed: 0.07,
            goal_position: 0.5,
            goal_velocity: 0.0,
            force: 0.001,
            gravity: 0.0025,
            state: MountainCarObservation::default(),
            dist: Uniform::from(-0.6..-0.4),
        };
        env.state = env.initialize_car();
        env
    }

    fn initialize_car(&self) -> MountainCarObservation {
        MountainCarObservation {
            position: self.dist.sample(&mut rand::thread_rng()),
            velocity: 0.0,
        }
    }
}

impl Default for MountainCarEnv {
    fn default() -> Self {
        Self::new(500)
    }
}

impl Env<MountainCarObservation, 3> for MountainCarEnv {
    fn reset(&mut self) -> MountainCarObservation {
        self.state = self.initialize_car();
        self.ready = true;
        self.curr_step = 0;
        self.state.clone()
    }

    fn step(&mut self, action: usize) -> Result<(MountainCarObservation, f64, bool), EnvNotReady> {
        if !self.ready {
            return Err(EnvNotReady);
        }
        if self.curr_step >= self.max_steps {
            self.ready = false;
            return Ok((self.state.clone(), -1.0, true));
        }
        self.curr_step += 1;

        self.state.velocity +=
            (action - 1) as f32 * self.force + (3.0 * self.state.position).cos() * (-self.gravity);
        self.state.velocity = self.state.velocity.clamp(-self.max_speed, self.max_speed);
        self.state.position += self.state.velocity;
        self.state.position = self
            .state
            .position
            .clamp(self.min_position, self.max_position);
        if self.state.position == self.min_position && self.state.velocity < 0.0 {
            self.state.velocity = 0.0
        }
        let terminated =
            self.state.position >= self.goal_position && self.state.velocity >= self.goal_velocity;
        if terminated {
            self.ready = false;
        }
        Ok((self.state.clone(), -1.0, terminated))
    }

    fn render(&self) -> String {
        "TODO".to_string()
    }

    fn get_action_label(&self, action: usize) -> &str {
        Self::ACTIONS[action]
    }
}
