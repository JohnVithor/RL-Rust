// use burn::{module::Module, tensor::backend::Backend};
// use environments::env::{Action, DiscreteAction};
// use std::hash::Hash;

// use self::network::Network;
// mod network;

// #[derive(Debug, Clone)]
// pub struct DiscreteNeuralPolicy<'a, B: Backend, T: Hash + PartialEq + Eq + Clone, A: DiscreteAction>
// where
//     [(); A::RANGE]: Sized,
// {
//     network: &'a dyn Network<B, 3, { A::RANGE }>,
// }
