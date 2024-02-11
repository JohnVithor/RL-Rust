#[derive(PartialEq, Eq)]
pub enum SpaceType {
    Discrete,
    Continuous,
    Mixed,
}

pub enum SpaceTypeBounds {
    Discrete(usize),
    Continuous(f64, f64),
}

pub struct SpaceInfo {
    pub data: Vec<SpaceTypeBounds>,
}

impl SpaceInfo {
    pub fn new(data: Vec<SpaceTypeBounds>) -> Self {
        Self { data }
    }
    pub fn get_type(&self) -> SpaceType {
        let mut is_discrete = false;
        let mut is_continuous = false;
        for i in self.data.iter() {
            match *i {
                SpaceTypeBounds::Discrete(_) => {
                    is_discrete = true;
                }
                SpaceTypeBounds::Continuous(_, _) => {
                    is_continuous = true;
                }
            };
        }
        if is_discrete && !is_continuous {
            SpaceType::Discrete
        } else if !is_discrete && is_continuous {
            SpaceType::Continuous
        } else if is_discrete && is_continuous {
            SpaceType::Mixed
        } else {
            panic!("Empty space")
        }
    }
    pub fn get_discrete_combinations(&self) -> usize {
        let mut value = 1;
        for i in self.data.iter() {
            match *i {
                SpaceTypeBounds::Discrete(n) => {
                    value *= n;
                }
                SpaceTypeBounds::Continuous(_, _) => {}
            };
        }
        value
    }
}
