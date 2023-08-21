// use burn::{
//     module::Module,
//     nn::{self, loss::CrossEntropyLoss, BatchNorm, PaddingConfig2d},
//     tensor::{
//         backend::{ADBackend, Backend},
//         Tensor,
//     },
//     train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
// };
// use std::fmt::Debug;

// pub trait Network<B: Backend, const I: usize, const O: usize>: Debug {
//     fn predict(&self, input: Tensor<B, I>);
//     fn update(&mut self, input: Tensor<B, I>, expected_putbput: Tensor<B, O>);
// }
