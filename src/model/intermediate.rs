use crate::Network;

#[derive(Debug)]
pub(super) struct LayerGradient {
    pub weights: Vec<f64>,
    pub biases: Vec<f64>,
}

#[derive(Debug)]
pub(super) struct IntermediateCache {
    pub fw_cache: Vec<Vec<f64>>,
    pub deltas: Vec<Vec<f64>>,
    pub layer_grads: Vec<LayerGradient>,
}

impl IntermediateCache {
    pub(super) fn new(network: &Network) -> Self {
        let layer_len = network.layers.len();
        let mut fw_cache = Vec::with_capacity(layer_len);
        let mut deltas = Vec::with_capacity(layer_len);
        let mut layer_grads = Vec::with_capacity(layer_len);

        for layer in network.layers.iter() {
            fw_cache.push(Vec::with_capacity(layer.neurons));
            deltas.push(Vec::with_capacity(layer.neurons));
            layer_grads.push(LayerGradient {
                weights: vec![0.0; layer.neurons * layer.inputs],
                biases: vec![0.0; layer.neurons],
            });
        }

        Self {
            fw_cache,
            deltas,
            layer_grads,
        }
    }

    pub(super) fn clear_inner(&mut self) {
        self.fw_cache.iter_mut().for_each(|v| v.clear());
        self.deltas.iter_mut().for_each(|v| v.clear());
    }

    pub(super) fn reset_grads(&mut self) {
        self.layer_grads.iter_mut().for_each(|grad| {
            grad.weights.fill(0.0);
            grad.biases.fill(0.0);
        });
    }
}
