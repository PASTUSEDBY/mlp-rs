use super::strategy::Activation;
use binrw::binrw;

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

    #[br(count = neurons * inputs)]
    pub matrix: Vec<f64>, // flat array, rows x columns

    #[br(count = neurons)]
    pub biases: Vec<f64>, // rows len
}

impl Layer {
    #[inline]
    pub fn weight(&self, neuron: usize, input: usize) -> f64 {
        self.matrix[neuron * self.inputs + input]
    }
}
