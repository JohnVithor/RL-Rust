pub fn save_json(path: &str, data: serde_json::Value) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    serde_json::to_writer(&mut file, &data)?;
    Ok(())
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
