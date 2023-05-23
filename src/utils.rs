pub fn argmax<T: PartialOrd>(vec: &Vec<T>) -> usize {
    return vec
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(index, _)| index)
        .unwrap_or_default();
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
