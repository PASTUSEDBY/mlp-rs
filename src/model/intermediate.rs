use crate::Network;

pub(super) struct IntermediateCache {
    pub(super) fw_cache: Vec<Vec<f64>>,
    pub(super) deltas: Vec<Vec<f64>>,
}

impl IntermediateCache {
    pub(super) fn new(network: &Network) -> Self {
        let layer_len = network.layers.len();
        let mut fw_cache = Vec::with_capacity(layer_len);
        let mut deltas = Vec::with_capacity(layer_len);

        for layer in network.layers.iter() {
            fw_cache.push(Vec::with_capacity(layer.neurons));
            deltas.push(Vec::with_capacity(layer.neurons));
        }

        Self { fw_cache, deltas }
    }

    pub(super) fn clear_inner(&mut self) {
        self.fw_cache.iter_mut().for_each(|v| v.clear());
        self.deltas.iter_mut().for_each(|v| v.clear());
    }
}
