mod concurrency;
mod sequential;

use threadpool::ThreadPool;

use super::{
    Layer, ModelError, Network,
    intermediate::{IntermediateCache, LayerGradient},
    strategy::{Activation, ExecutionStrategy, Loss},
};

const POOL_EXHAUSTED: &str = "The workspace buffer pool was unexpectedly exhausted! Looks like a tracking mismatch between parent and workers.";
const ARC_MUT_FAIL_NETWORK: &str = "Failed to grab a unique mutable reference to the network layers. A worker thread is likely holding one.";
const ARC_MUT_FAIL_INDICES: &str = "Failed to grab a unique mutable reference to the indices to shuffle. A worker thread is likely holding one.";
const CHANNEL_BROKEN: &str = "Failed to send the intermediate cache back to the main thread, the receiver might have been already dropped.";

#[derive(Debug)]
pub struct Optimizer {
    pub learning_rate: f64,
    pub batch_size: usize,
    pub loss: Loss,
    pub strategy: ExecutionStrategy,
    pool: Option<ThreadPool>,
}

impl Optimizer {
    pub fn new(
        learning_rate: f64,
        batch_size: usize,
        loss: Loss,
        strategy: ExecutionStrategy,
    ) -> Self {
        let pool = match &strategy {
            ExecutionStrategy::Sequential => None,
            ExecutionStrategy::Concurrent { workers } => {
                Some(workers.map(ThreadPool::new).unwrap_or_default())
            }
        };

        Self {
            learning_rate,
            batch_size,
            loss,
            strategy,
            pool,
        }
    }

    pub fn train(
        &self,
        network: &mut Network,
        inputs: Vec<Vec<f64>>,
        exps: Vec<Vec<f64>>,
        epochs: usize,
    ) -> Result<(), ModelError> {
        self.train_impl::<fn(usize) -> ()>(network, inputs, exps, epochs, None)
    }

    pub fn train_epoch_handler<F: Fn(usize) -> ()>(
        &self,
        network: &mut Network,
        inputs: Vec<Vec<f64>>,
        exps: Vec<Vec<f64>>,
        epochs: usize,
        on_epochs_finish: F,
    ) -> Result<(), ModelError> {
        self.train_impl(network, inputs, exps, epochs, Some(on_epochs_finish))
    }

    fn train_impl<F: Fn(usize) -> ()>(
        &self,
        network: &mut Network,
        inputs: Vec<Vec<f64>>,
        exps: Vec<Vec<f64>>,
        epochs: usize,
        on_epoch_finish: Option<F>,
    ) -> Result<(), ModelError> {
        self.validate(network, &inputs, &exps)?;

        let executor = match &self.strategy {
            ExecutionStrategy::Sequential => Self::train_sequential,
            ExecutionStrategy::Concurrent { workers: _ } => Self::train_concurrent,
        };

        executor(self, network, inputs, exps, epochs, on_epoch_finish);

        Ok(())
    }

    fn validate(
        &self,
        network: &Network,
        inputs: &Vec<Vec<f64>>,
        exps: &Vec<Vec<f64>>,
    ) -> Result<(), ModelError> {
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

        Ok(())
    }

    fn propagate_batch(
        net: &Network,
        loss: Loss,
        inps: &Vec<Vec<f64>>,
        exps: &Vec<Vec<f64>>,
        inp_indices: &[usize],
        im: &mut IntermediateCache,
    ) {
        let last_layer = net.layers.len() - 1;
        for &inp_idx in inp_indices {
            let (input, expected) = (&inps[inp_idx], &exps[inp_idx]);

            im.clear_inner(); // clear previous cache
            net.forward(input, &mut im.fw_cache);
            loss.output_delta(
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
                let grad = &mut im.layer_grads[l_idx];
                Self::layer_gradients(layer, layer_input, delta, grad);
            }
        }
    }

    fn update_batch(&self, layers: &mut [Layer], layer_grads: &[LayerGradient], batch_size: f64) {
        let adjustment = self.learning_rate / batch_size;
        for (layer, grad) in layers.iter_mut().zip(layer_grads) {
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
