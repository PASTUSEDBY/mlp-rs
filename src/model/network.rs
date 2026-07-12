use std::{
    io::{Read, Seek, Write},
    mem,
    sync::Arc,
};

use super::{ModelError, intermediate::IntermediateCache, layer::Layer, strategy::Activation};

use binrw::{BinReaderExt, BinWrite, binrw};
use rand::rng as rngfn;
use rand_distr::{Distribution, Normal};

#[binrw]
#[brw(little, magic = b"\0MLP_RS\n")]
#[derive(Debug, Clone)]
pub struct Network {
    // this field is temporary and only used by the binrw crate
    // to pack the length and parse it when unpacking
    #[br(temp)]
    #[bw(calc = layers.len() as u64)]
    layer_len: u64,
    // even though the minimum size must be 2 theoretically,
    // a Layer here defines the layers trainable (contains weights)
    // while the input layer is a simple f64 slice of a vector like object
    #[br(count = layer_len as usize, map = Vec::into)]
    #[bw(map = Arc::as_ref)]
    pub layers: Arc<[Layer]>,
}

impl Network {
    pub fn new(layers: &[usize], actvs: &[Activation]) -> Result<Self, ModelError> {
        if layers.len() < 2 {
            return Err(ModelError::MinLayerSize(2, layers.len()));
        }

        if actvs.len() != layers.len() - 1 {
            return Err(ModelError::ActivationLayerMismatch(
                layers.len() - 1,
                actvs.len(),
            ));
        }

        let mut rng = rngfn();
        let mut network_layers = vec![];

        for (&[inputs, neurons], actv) in layers.array_windows::<2>().zip(actvs) {
            let std_dev = actv.std_dev(inputs, neurons);
            let normal = Normal::new(0.0, std_dev)?.sample_iter(&mut rng);

            let matrix = normal.take(inputs * neurons).collect::<Vec<_>>().into();
            let biases = vec![0.0; neurons].into();

            network_layers.push(Layer {
                activation: actv.clone(),
                neurons,
                inputs,
                matrix,
                biases,
            });
        }

        Ok(Network {
            layers: network_layers.into(),
        })
    }

    pub(super) fn intermediate_cache(&self) -> IntermediateCache {
        IntermediateCache::new(self)
    }

    pub fn save<W: Write + Seek>(&self, dest: &mut W) -> Result<(), ModelError> {
        self.write_le(dest)?;
        Ok(())
    }

    pub fn load<R: Read + Seek>(src: &mut R) -> Result<Self, ModelError> {
        let network = src.read_le()?;
        Ok(network)
    }

    // inference logic
    // i could've used forward(), but nop, inefficient
    // gotta keep the dup logic
    pub fn predict(&self, inputs: &[f64]) -> Vec<f64> {
        let mut curr_in = inputs.to_vec();
        let mut this_out = vec![];

        for layer in self.layers.iter() {
            let raw_scores = layer
                .matrix
                .chunks_exact(layer.inputs)
                .zip(layer.biases.iter())
                .map(|(chunk, &bias)| {
                    chunk
                        .iter()
                        .zip(&curr_in)
                        .fold(bias, |acc, (&w, &x)| acc + w * x)
                });

            layer.activation.apply(raw_scores, &mut this_out);
            mem::swap(&mut curr_in, &mut this_out);
            this_out.clear();
        }

        curr_in
    }

    // returns the activation cache for each neuron
    // for now, the activation value is stored cuz the derivative is easily computable from it
    pub(super) fn forward(&self, inputs: &[f64], fw_cache: &mut Vec<Vec<f64>>) {
        let mut curr_in = inputs;
        for (layer, cache) in self.layers.iter().zip(fw_cache) {
            let raw_scores = layer
                .matrix
                .chunks_exact(layer.inputs)
                .zip(layer.biases.iter())
                .map(|(chunk, &bias)| {
                    chunk
                        .iter()
                        .zip(curr_in)
                        .fold(bias, |acc, (&w, &x)| acc + w * x)
                });

            layer.activation.apply(raw_scores, cache);
            curr_in = cache;
        }
    }

    // returns the delta vectors from the backpropagation, in order
    pub(super) fn backward(&self, fw_cache: &[Vec<f64>], deltas: &mut Vec<Vec<f64>>) {
        for layer_idx in (0..self.layers.len() - 1).rev() {
            let layer = &self.layers[layer_idx];
            let next_layer = &self.layers[layer_idx + 1];
            let hidden_out = &fw_cache[layer_idx];
            // we run into an annoying borrow checker issue here, so what we do is
            // explicitly prove that they are disjoint
            let (left, right) = deltas.split_at_mut(layer_idx + 1);
            let delta_next = &right[0];
            let delta_curr = &mut left[layer_idx];

            delta_curr.extend(hidden_out.iter().enumerate().map(|(curr_neuron, &h_out)| {
                let error = delta_next
                    .iter()
                    .enumerate()
                    .map(|(next_neuron, &delta)| {
                        next_layer.weight(next_neuron, curr_neuron) * delta
                    })
                    .sum::<f64>();

                error * layer.activation.derivative_with_output(h_out)
            }));
        }
    }
}
