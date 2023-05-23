#[derive(Hash, Debug, Clone, PartialEq, Eq, Ord, PartialOrd)]
pub struct Observation {
    pub p_score: u8,
    pub d_score: u8,
    pub p_ace: bool,
}

impl Observation {
    pub fn new(p_score: u8, d_score: u8, p_ace: bool) -> Self {
        return Self {
            p_score,
            d_score,
            p_ace,
        }
    }

    pub fn get_id(&self) -> u16 {
        let a: u8 = if self.p_score <= 21 { self.p_score-3 } else {22};
        let b: u8 = if self.d_score <= 11 { self.d_score-1 } else {12};
        let value: u8 = if a >=b {a * a + a + b} else {a + b * b};
        if self.p_ace {
            return value as u16 + 347;
        } else {
            return (value - 3 ) as u16;
        }
    }
}