pub fn argmax<T: PartialOrd>(vec: &Vec<T>) -> usize {
    let mut max: &T = &vec[0];
    let mut result: usize = 0;
    let mut i: usize = 0;
    for v in vec {
        if v > &max {
            max = v;
            result = i;
        }
        i+=1;
    }
    return result;
}

pub fn categorical_sample(probs: &Vec<f64>, random: f64) -> usize {
    let mut b: f64 = 0.0;
    let r: Vec<bool> = probs.iter().map(|a| {
        b+=a;
        b > random
    }).collect();
    return argmax(&r);
}

pub fn moving_average(window: usize, vector: &Vec<f64>) -> Vec<f64> {
    let mut aux: usize = 0;
    let mut result: Vec<f64> = vec![];
    while aux < vector.len() {
        let end: usize = if aux + window < vector.len() {
            aux + window
        } else {
            vector.len()
        };
        let slice: &[f64] = &vector[aux..end];
        let r: f64 = slice.iter().sum();
        result.push(r / window as f64);
        aux = end;
    }
    return result;
}
