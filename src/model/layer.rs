use super::strategy::Activation;
use binrw::binrw;
use std::sync::Arc;

#[binrw]
#[brw(little)]
#[derive(Debug)]
pub struct Layer {
    pub activation: Activation, // the activation strategy

    #[br(map = |x: u64| x as usize)]
    #[bw(map = |&x| x as u64)]
    pub neurons: usize, // no. of neurons (rows)

    #[br(map = |x: u64| x as usize)]
    #[bw(map = |&x| x as u64)]
    pub inputs: usize, // the input weights (cols)

    #[br(count = inputs * neurons, map = Vec::into)]
    #[bw(map = Arc::as_ref)]
    pub(super) matrix: Arc<[f64]>, // flat array, rows x columns

    #[br(count = neurons, map = Vec::into)]
    #[bw(map = Arc::as_ref)]
    pub(super) biases: Arc<[f64]>, // rows len
}

impl Layer {
    pub fn matrix(&self) -> &[f64] {
        &self.matrix
    }

    pub fn biases(&self) -> &[f64] {
        &self.biases
    }

    #[inline]
    pub fn weight(&self, neuron: usize, input: usize) -> f64 {
        self.matrix[neuron * self.inputs + input]
    }
}
