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

use plotters::prelude::*;

pub fn plot_moving_average(
    values: &[Vec<f64>],
    colors: &[&RGBColor],
    legends: &[&str],
    title: &str,
) {
    let filename = format!("{}.png", title);
    let root_area = BitMapBackend::new(&filename, (600, 400)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let mut max_len = values[0].len();
    let mut min_value = values[0].iter().copied().reduce(f64::min).unwrap();
    let mut max_value = values[0].iter().copied().reduce(f64::max).unwrap();
    for vals in values.iter().skip(1) {
        if vals.len() > max_len {
            max_len = vals.len();
        }
        let min_value_i = vals.iter().copied().reduce(f64::min).unwrap();
        if min_value_i < min_value {
            min_value = min_value_i;
        }
        let max_value_i = vals.iter().copied().reduce(f64::max).unwrap();
        if max_value_i > max_value {
            max_value = max_value_i;
        }
    }

    println!("max len {}", max_len);
    println!("{} | {}\n", min_value, max_value);

    if min_value == max_value || min_value.is_nan() || max_value.is_nan() {
        min_value = -1.0;
        max_value = 1.0;
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
            (0..).zip(values[i].iter()).map(|(idx, y)| (idx, *y)),
            c,
        ))
        .unwrap()
        .label(legends[i])
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], *c));
    }

    ctx.configure_series_labels()
        .border_style(BLACK)
        .background_style(WHITE.mix(0.8))
        .draw()
        .unwrap();
}
