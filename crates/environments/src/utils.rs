#[inline(always)]
pub fn from_2d_to_1d(ncol: usize, row: usize, col: usize) -> usize {
    row * ncol + col
}

#[inline(always)]
pub fn from_1d_to_2d(ncol: usize, pos: usize) -> (usize, usize) {
    (pos / ncol, pos % ncol)
}

#[inline(always)]
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
