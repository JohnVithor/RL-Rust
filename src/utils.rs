use std::cmp::{max, min};

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

pub fn to_s(ncol: usize, row: usize, col: usize) -> usize{
    return row * ncol + col;
}

pub fn inc(nrow: usize, ncol: usize, row: usize, col: usize, a: usize) -> (usize, usize) {
    let new_col: usize;
    let new_row: usize;
    if a == 0 { // left
        new_col = if col != 0 { max(col - 1, 0) } else {0};
        new_row = row;
    } else if a == 1 { // down
        new_col = col;
        new_row = min(row + 1, nrow - 1);
    } else if a == 2 { // right
        new_col = min(col + 1, ncol - 1);
        new_row = row;
    } else if a == 3 { // up
        new_col = col;
        new_row = if row != 0 { max(row - 1, 0) } else {0};
    } else {
        return (row, col)
    }
    return (new_row, new_col)
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

use plotters::prelude::*;

pub fn plot_moving_average(values: &Vec<&Vec<f64>>,
                    colors: &Vec<&RGBColor>,
                    legends: &Vec<&str>,
                    title: &str) {

    let filename = format!("{}.png", title);
    let root_area = BitMapBackend::new(&filename, (600, 400)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let mut max_len = values[0].len();
    let mut min_value = values[0].iter().copied().reduce(f64::min).unwrap();
    let mut max_value = values[0].iter().copied().reduce(f64::max).unwrap();
    for i in 1..values.len() {
        if values[i].len() > max_len {
            max_len = values[i].len();
        }
        let min_value_i = values[0].iter().copied().reduce(f64::min).unwrap();
        if min_value_i < min_value {
            min_value = min_value_i;
        }
        let max_value_i = values[0].iter().copied().reduce(f64::max).unwrap();
        if max_value_i > max_value {
            max_value = max_value_i;
        }
    }

    let mut ctx = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption(title, ("sans-serif", 40))
        .build_cartesian_2d(0..max_len, min_value..max_value)
        .unwrap();

    ctx.configure_mesh().draw().unwrap();

    for i in 0..values.len() {
        let c = colors[i];
        ctx.draw_series(LineSeries::new(
            (0..)
                .zip(values[i].iter())
                .map(|(idx, y)| (idx, *y)),
            c,
        ))
        .unwrap()
        .label(legends[i])
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], c.clone()));
    }
    
    ctx.configure_series_labels()
        .border_style(&BLACK)
        .background_style(&WHITE.mix(0.8))
        .draw()
        .unwrap();
    
}