use ndarray::Axis;
use tch::Tensor;

fn main() {
    let a = Tensor::randn([2, 3], tch::kind::FLOAT_CPU);
    let b = Tensor::randn([2, 3], tch::kind::FLOAT_CPU);
    println!("a: {}", a);
    println!("b: {}", b);
    let a: ndarray::ArrayD<f32> = (&a).try_into().unwrap();
    let b: ndarray::ArrayD<f32> = (&b).try_into().unwrap();
    // println!("b: {}", b);
    a.axis_iter(Axis(0))
        .zip(b.axis_iter(Axis(0)))
        .for_each(|(i, j)| {
            println!("a: {}, b: {}", i, j);
        })
}
