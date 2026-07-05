use thiserror::Error;

mod layer;
mod network;
mod optimizer;
pub mod strategy;

use binrw::Error as BWError;
use rand_distr::NormalError;
use std::fmt::Debug;
use std::io::Error as IOError;
use strategy::{Activation, Loss};

// re exports for convenience
#[allow(unused_imports)]
pub use layer::Layer;
pub use network::Network;
pub use optimizer::Optimizer;

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("input layer size must be atleast {0}, received: {1}")]
    MinLayerSize(usize, usize),
    #[error("activation length must be one less than input layer length ({lay_len}), expected: {0}, got: {1}", lay_len = .0 + 1)]
    ActivationLayerMismatch(usize, usize),
    #[error("activation function {0:?} is not compatible with the loss function {1:?}")]
    BadCombination(Activation, Loss),
    #[error(transparent)]
    BadNormalParameters(#[from] NormalError),
    #[error(transparent)]
    BinWriteError(#[from] BWError),
    #[error(transparent)]
    IOError(#[from] IOError),
}
