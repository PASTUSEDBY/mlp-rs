use super::Optimizer;
use crate::{
    Network,
    model::optimizer::{
        ARC_MUT_FAIL_INDICES, ARC_MUT_FAIL_NETWORK, CHANNEL_BROKEN, POOL_EXHAUSTED,
    },
};
use rand::{rng as rngfn, seq::SliceRandom};
use std::sync::{Arc, mpsc};

impl Optimizer {
    pub(super) fn train_concurrent<F: Fn(usize) -> ()>(
        &self,
        network: &mut Network,
        inputs: Vec<Vec<f64>>,
        exps: Vec<Vec<f64>>,
        epochs: usize,
        on_epochs_finish: Option<F>,
    ) {
        let pool = self.pool.as_ref().unwrap();
        let num_workers = pool.max_count();

        let (grad_tx, grad_rx) = mpsc::channel();
        let mut buffer_pool = (0..num_workers)
            .map(|_| network.intermediate_cache())
            .collect::<Vec<_>>();

        let mut rng = rngfn();
        let mut indices: Arc<[usize]> = (0..inputs.len()).collect::<Vec<_>>().into();
        let inputs = Arc::new(inputs);
        let exps = Arc::new(exps);
        let mut main_im_cache = network.intermediate_cache(); // we allocate extra vecs for now, but we want layer grads only

        for epoch in 0..epochs {
            Arc::get_mut(&mut indices)
                .expect(ARC_MUT_FAIL_INDICES)
                .shuffle(&mut rng);

            let mut batch_start_idx = 0usize;
            for inp_indices in indices.chunks(self.batch_size) {
                main_im_cache.reset_grads();
                let curr_size = inp_indices.len();
                let calc_chunk_size = curr_size / num_workers;
                let remainder = curr_size % num_workers;
                let mut active_workers = 0usize;

                for w_idx in 0..num_workers {
                    let worker_chunk_size = calc_chunk_size + (w_idx < remainder) as usize;
                    if worker_chunk_size == 0 {
                        continue; // skip this empty workload
                    }
                    active_workers += 1;

                    let start = batch_start_idx;
                    let end = start + worker_chunk_size;
                    batch_start_idx = end;

                    let net_clone = network.clone();
                    let loss = self.loss.clone();
                    let grad_tx = grad_tx.clone();
                    let inputs = Arc::clone(&inputs);
                    let exps = Arc::clone(&exps);
                    let indices = Arc::clone(&indices);
                    let mut im_cache = buffer_pool.pop().expect(POOL_EXHAUSTED);

                    pool.execute(move || {
                        let worker_indices = &indices[start..end];
                        im_cache.reset_grads();
                        Self::propagate_batch(
                            &net_clone,
                            loss,
                            &inputs,
                            &exps,
                            worker_indices,
                            &mut im_cache,
                        );

                        // to avoid race conditions, we do a "manual sync" here
                        // this establishes that the Arcs are dropped before the main thread receives any update
                        drop(net_clone);
                        drop(indices);
                        grad_tx.send(im_cache).expect(CHANNEL_BROKEN);
                    });
                }

                // now we collect in the main cache and then update
                for im in grad_rx.iter().take(active_workers) {
                    for (m, w) in main_im_cache.layer_grads.iter_mut().zip(&im.layer_grads) {
                        m.weights
                            .iter_mut()
                            .zip(&w.weights)
                            .for_each(|(x, y)| *x += y);
                        m.biases
                            .iter_mut()
                            .zip(&w.biases)
                            .for_each(|(x, y)| *x += y);
                    }

                    // add back to the pool
                    buffer_pool.push(im);
                }

                let layers = Arc::get_mut(&mut network.layers).expect(ARC_MUT_FAIL_NETWORK);
                self.update_batch(layers, &main_im_cache.layer_grads, inp_indices.len() as f64);
            }

            if let Some(ref func) = on_epochs_finish {
                func(epoch);
            };
        }
    }
}
