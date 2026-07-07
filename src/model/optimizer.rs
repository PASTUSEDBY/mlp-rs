use crate::model::{ModelError, intermediate::IntermediateCache, strategy::Activation};

use super::{Layer, Network, strategy::Loss};
use rand::{rng as rngfn, seq::SliceRandom};

#[derive(Debug)]
pub struct Optimizer {
    pub learning_rate: f64,
    pub batch_size: usize,
    pub loss: Loss,
}

#[derive(Debug)]
struct LayerGradient {
    weights: Vec<f64>,
    biases: Vec<f64>,
}

impl Optimizer {
    pub fn new(learning_rate: f64, batch_size: usize, loss: Loss) -> Self {
        Self {
            learning_rate,
            batch_size,
            loss,
        }
    }

    pub fn train(
        &self,
        network: &mut Network,
        inputs: &[&[f64]],
        exps: &[&[f64]],
        epochs: usize,
    ) -> Result<(), ModelError> {
        self.train_impl(network, inputs, exps, epochs, false)
    }

    pub fn train_verbose(
        &self,
        network: &mut Network,
        inputs: &[&[f64]],
        exps: &[&[f64]],
        epochs: usize,
    ) -> Result<(), ModelError> {
        self.train_impl(network, inputs, exps, epochs, true)
    }

    fn train_impl(
        &self,
        network: &mut Network,
        inputs: &[&[f64]],
        exps: &[&[f64]],
        epochs: usize,
        verbose: bool, // to print progress to stderr for now
    ) -> Result<(), ModelError> {
        // only checking the outer length, upto the caller if they don't provide formatted data
        if inputs.len() != exps.len() {
            return Err(ModelError::InputExpectedMismatch(inputs.len(), exps.len()));
        }

        // we need some checks for cross entropy loss
        if let Loss::CrossEntropy = self.loss {
            let last = network.layers.last().unwrap(); // should be safe
            if !matches!(last.activation, Activation::SoftMax | Activation::Sigmoid) {
                return Err(ModelError::BadCombination(
                    last.activation.clone(),
                    self.loss.clone(),
                ));
            }
        }

        let mut rng = rngfn();
        let mut indices: Vec<usize> = (0..inputs.len()).collect();
        let mut grads: Vec<LayerGradient> = network
            .layers
            .iter()
            .map(|layer| LayerGradient {
                weights: vec![0.0; layer.neurons * layer.inputs],
                biases: vec![0.0; layer.neurons],
            })
            .collect();

        let mut im = network.intermediate_cache();
        for epoch in 0..epochs {
            // shuffle the input and feed it
            if verbose {
                eprintln!("In Epoch {} right now.", epoch + 1);
            }
            indices.shuffle(&mut rng);
            for inp_indices in indices.chunks(self.batch_size) {
                // and let's zero the grads cache out
                for grad in grads.iter_mut() {
                    grad.weights.fill(0.0);
                    grad.biases.fill(0.0);
                }
                self.train_batch(network, inputs, exps, inp_indices, &mut grads, &mut im);
            }
        }

        Ok(())
    }

    fn train_batch(
        &self,
        net: &mut Network,
        inps: &[&[f64]],
        exps: &[&[f64]],
        inp_indices: &[usize],
        grads: &mut [LayerGradient],
        im: &mut IntermediateCache,
    ) {
        let last_layer = net.layers.len() - 1;
        for &inp_idx in inp_indices {
            let (input, expected) = (inps[inp_idx], exps[inp_idx]);

            im.clear_inner(); // clear previous cache
            net.forward(input, &mut im.fw_cache);
            self.loss.output_delta(
                &net.layers[last_layer].activation,
                &im.fw_cache[last_layer],
                expected,
                &mut im.deltas[last_layer],
            );
            net.backward(&im.fw_cache, &mut im.deltas);

            for (l_idx, layer) in net.layers.iter().enumerate() {
                let layer_input = if l_idx == 0 {
                    input
                } else {
                    &im.fw_cache[l_idx - 1]
                };
                let delta = &im.deltas[l_idx];
                let grad = &mut grads[l_idx];
                self.layer_gradients(layer, layer_input, delta, grad);
            }
        }

        // now we update with the average weights and biases gradient
        let batch_size = inp_indices.len() as f64;
        let adjustment = self.learning_rate / batch_size;
        for (layer, grad) in net.layers.iter_mut().zip(grads) {
            layer
                .matrix
                .iter_mut()
                .zip(&grad.weights)
                .for_each(|(w, g)| *w -= g * adjustment);
            layer
                .biases
                .iter_mut()
                .zip(&grad.biases)
                .for_each(|(b, g)| *b -= g * adjustment);
        }
    }

    fn layer_gradients(
        &self,
        layer: &Layer,
        layer_input: &[f64],
        delta: &[f64],
        grad: &mut LayerGradient,
    ) {
        for ((neuron_row, grad_bias), &neuron_delta) in grad
            .weights
            .chunks_exact_mut(layer.inputs)
            .zip(&mut grad.biases)
            .zip(delta)
        {
            for (grad_wt, &x) in neuron_row.iter_mut().zip(layer_input) {
                *grad_wt += neuron_delta * x;
            }

            *grad_bias += neuron_delta;
        }
    }
}
