use super::Optimizer;
use crate::{Network, model::optimizer::ARC_MUT_FAIL_NETWORK};
use rand::{rng as rngfn, seq::SliceRandom};
use std::sync::Arc;

impl Optimizer {
    pub(super) fn train_sequential<F: Fn(usize) -> ()>(
        &self,
        network: &mut Network,
        inputs: Vec<Vec<f64>>,
        exps: Vec<Vec<f64>>,
        epochs: usize,
        on_epochs_finish: Option<F>,
    ) {
        let mut rng = rngfn();
        let mut indices: Vec<usize> = (0..inputs.len()).collect();

        let mut im = network.intermediate_cache();
        for epoch in 0..epochs {
            // shuffle the input and feed it
            indices.shuffle(&mut rng);
            for inp_indices in indices.chunks(self.batch_size) {
                // and let's zero the grads cache out
                im.reset_grads();
                Self::propagate_batch(
                    network,
                    self.loss.clone(),
                    &inputs,
                    &exps,
                    inp_indices,
                    &mut im,
                );

                // now we update with the average weights and biases gradient
                self.update_batch(
                    Arc::get_mut(&mut network.layers).expect(ARC_MUT_FAIL_NETWORK), // altho this should never happen
                    &im.layer_grads,
                    inp_indices.len() as f64,
                );
            }

            if let Some(ref func) = on_epochs_finish {
                func(epoch);
            };
        }
    }
}
