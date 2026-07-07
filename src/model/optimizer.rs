use crate::model::{ModelError, strategy::Activation};

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
                self.train_batch(network, inputs, exps, inp_indices, &mut grads);
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
    ) {
        for &inp_idx in inp_indices {
            // just to remove the bounds checks cuz we know it's safe
            let (input, expected) =
                unsafe { (*inps.get_unchecked(inp_idx), *exps.get_unchecked(inp_idx)) };
            let fw_cache = net.forward(input);
            // following unwraps are safe
            let predicted = fw_cache.last().unwrap();
            let delta_out =
                self.loss
                    .output_delta(&net.layers.last().unwrap().activation, predicted, expected);
            let deltas = net.backward(&fw_cache, delta_out);

            for (l_idx, layer) in net.layers.iter().enumerate() {
                let layer_input = if l_idx == 0 {
                    input
                } else {
                    &fw_cache[l_idx - 1]
                };
                let delta = &deltas[l_idx];
                let grad = &mut grads[l_idx];
                self.layer_gradients(layer, layer_input, delta, grad);
            }
        }

        // now we update with the average weights and biases gradient
        let batch_size = inp_indices.len() as f64;
        for (layer, grad) in net.layers.iter_mut().zip(grads) {
            layer
                .matrix
                .iter_mut()
                .zip(&grad.weights)
                .for_each(|(w, g)| *w -= self.learning_rate * g / batch_size);
            layer
                .biases
                .iter_mut()
                .zip(&grad.biases)
                .for_each(|(b, g)| *b -= self.learning_rate * g / batch_size);
        }
    }

    fn layer_gradients(
        &self,
        layer: &Layer,
        layer_input: &[f64],
        delta: &[f64],
        grad: &mut LayerGradient,
    ) {
        for ((neuron_row, grad_bias), &gradient) in grad
            .weights
            .chunks_exact_mut(layer.inputs)
            .zip(&mut grad.biases)
            .zip(delta)
        {
            for (grad_wt, &x) in neuron_row.iter_mut().zip(layer_input) {
                *grad_wt += gradient * x;
            }

            *grad_bias += gradient;
        }
    }
}
