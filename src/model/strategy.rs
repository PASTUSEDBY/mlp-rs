use binrw::binrw;

#[binrw]
#[brw(little)]
#[derive(Debug, Clone)]
pub enum Activation {
    #[brw(magic(0u8))]
    Sigmoid,
    #[brw(magic(1u8))]
    LReLU(f64), // the parameter can be 0.0 to get normal ReLU
    #[brw(magic(2u8))]
    Linear,
    #[brw(magic(3u8))]
    SoftMax,
}

fn component_apply(xs: impl IntoIterator<Item = f64>, f: impl Fn(f64) -> f64, out: &mut Vec<f64>) {
    out.extend(xs.into_iter().map(f));
}

impl Activation {
    pub(super) fn apply(&self, xs: impl IntoIterator<Item = f64>, out: &mut Vec<f64>) {
        match self {
            Activation::Sigmoid => component_apply(xs, |x| 1.0 / (1.0 + (-x).exp()), out),
            Activation::LReLU(alpha) => component_apply(xs, |x| x.max(*alpha * x), out),
            Activation::Linear => component_apply(xs, std::convert::identity, out),
            Activation::SoftMax => {
                let xs: Vec<f64> = xs.into_iter().collect();
                // so the issue is, exp(M) where M is a big enough number, will overflow
                // we need to get the maximum element, and reduce it
                // won't affect the end answer cuz its a probability and the factor will cancel out
                let max = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);

                // we can calc the denominator with the numerators in one pass
                let mut exps = Vec::with_capacity(xs.len());
                let mut denominator = 0.0;
                for x in xs {
                    let exp = (x - max).exp();
                    denominator += exp;
                    exps.push(exp);
                }

                out.extend(exps.into_iter().map(|exp| exp / denominator));
            }
        }
    }

    pub(super) fn derivative_with_output(&self, y: f64) -> f64 {
        match self {
            Activation::Sigmoid => y * (1.0 - y),
            Activation::LReLU(alpha) => {
                if y > 0.0 {
                    1.0
                } else {
                    *alpha
                }
            }
            Activation::Linear => 1.0,
            Activation::SoftMax => todo!("can't be bothered with the Jacobian"),
        }
    }

    pub(super) fn std_dev(&self, input: usize, output: usize) -> f64 {
        let input = input as f64;
        let output = output as f64;

        match self {
            // Xavier initialization
            Activation::Sigmoid | Activation::Linear | Activation::SoftMax => {
                (2.0 / (input + output)).sqrt()
            }
            // He initialization
            Activation::LReLU(alpha) => {
                let factor = 1.0 + alpha.powi(2);
                (2.0 / (input * factor)).sqrt()
            }
        }
    }
}

#[derive(Debug, Clone)]
#[allow(unused)]
pub enum Loss {
    MeanSquare,
    CrossEntropy,
}

impl Loss {
    #[allow(unused)] // TODO: Calculate the losses and return it?
    // TODO: the `/ num` in MeanSquare might be problematic
    pub(super) fn loss(&self, actv: &Activation, predicted: &[f64], expected: &[f64]) -> f64 {
        let num = predicted.len() as f64;
        const EPSILON: f64 = 1e-10;

        predicted
            .iter()
            .zip(expected)
            .map(|(&o, &e)| match (actv, self) {
                (_, Loss::MeanSquare) => 0.5 * (o - e).powi(2) / num,
                (Activation::SoftMax, Loss::CrossEntropy) => -e * (o + EPSILON).ln(),
                (Activation::Sigmoid, Loss::CrossEntropy) => {
                    -(e * (o + EPSILON).ln() + (1.0 - e) * (1.0 - o + EPSILON).ln())
                }
                _ => unreachable!("this should never happen"),
            })
            .sum::<f64>()
    }

    pub(super) fn output_delta(
        &self,
        actv: &Activation,
        predicted: &[f64],
        expected: &[f64],
        out: &mut Vec<f64>,
    ) {
        let it = predicted
            .iter()
            .zip(expected)
            .map(|(&o, &e)| match (actv, self) {
                (Activation::SoftMax | Activation::Sigmoid, Loss::CrossEntropy) => o - e,
                (_, Loss::MeanSquare) => (o - e) * actv.derivative_with_output(o),
                _ => unreachable!("this should never happen"),
            });

        out.extend(it);
    }
}

#[derive(Debug)]
pub enum ExecutionStrategy {
    Sequential,
    Concurrent { workers: Option<usize> },
}
