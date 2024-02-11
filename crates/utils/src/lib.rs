#[inline(always)]
pub fn argmax<T: PartialOrd>(values: impl Iterator<Item = T>) -> usize {
    values
        .enumerate()
        .max_by(|x, y| PartialOrd::partial_cmp(&x.1, &y.1).unwrap())
        .unwrap()
        .0
}

#[inline(always)]
pub fn categorical_sample(probs: &[f64], random: f64) -> usize {
    let mut b: f64 = 0.0;
    let r = probs.iter().map(|a| {
        b += a;
        b > random
    });
    argmax(r)
}

#[inline(always)]
pub fn wrap(value: f32, min: f32, max: f32) -> f32 {
    let diff = max - min;
    let mut result = value;
    while result > max {
        result -= diff
    }
    while result < min {
        result += diff
    }
    result
}

#[inline(always)]
pub fn bound(value: f32, min: f32, max: f32) -> f32 {
    value.max(min).min(max)
}
