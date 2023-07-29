pub fn argmax<T: PartialOrd>(vec: &[T]) -> usize {
    let mut max: &T = &vec[0];
    let mut result: usize = 0;
    for (i, v) in vec.iter().enumerate() {
        if v > max {
            max = v;
            result = i;
        }
    }
    result
}

pub fn max<T: PartialOrd + Clone>(vec: &[T]) -> T {
    let mut max: &T = &vec[0];
    for v in vec {
        if v > max {
            max = v;
        }
    }
    max.clone()
}

pub fn ndarray_max<T: PartialOrd + Clone>(vec: &ndarray::Array2<T>) -> T {
    let mut max: &T = &vec[(0, 0)];
    for v in vec.iter() {
        if v > max {
            max = v;
        }
    }
    max.clone()
}

pub fn categorical_sample(probs: &[f64], random: f64) -> usize {
    let mut b: f64 = 0.0;
    let r: Vec<bool> = probs
        .iter()
        .map(|a| {
            b += a;
            b > random
        })
        .collect();
    argmax(&r)
}

pub fn from_2d_to_1d(ncol: usize, row: usize, col: usize) -> usize {
    row * ncol + col
}

pub fn from_1d_to_2d(ncol: usize, pos: usize) -> (usize, usize) {
    (pos / ncol, pos % ncol)
}

pub fn inc(nrow: usize, ncol: usize, row: usize, col: usize, a: usize) -> (usize, usize) {
    let new_col: usize;
    let new_row: usize;
    if a == 0 {
        // left
        new_col = if col != 0 { (col - 1).max(0) } else { 0 };
        new_row = row;
    } else if a == 1 {
        // down
        new_col = col;
        new_row = (row + 1).min(nrow - 1);
    } else if a == 2 {
        // right
        new_col = (col + 1).min(ncol - 1);
        new_row = row;
    } else if a == 3 {
        // up
        new_col = col;
        new_row = if row != 0 { (row - 1).max(0) } else { 0 };
    } else {
        return (row, col);
    }
    (new_row, new_col)
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
    result
}

pub fn save_json(path: &str, data: serde_json::Value) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    serde_json::to_writer(&mut file, &data)?;
    Ok(())
}
