use mlp::{Network, strategy::Activation};
use mlp::{Optimizer, strategy::Loss};

fn main() -> anyhow::Result<()> {
    const EPOCHS: usize = 100000;
    let inputs: &[&[f64]] = &[&[0.0, 0.0], &[0.0, 1.0], &[1.0, 0.0], &[1.0, 1.0]];
    let expected: &[&[f64]] = &[&[0.0], &[1.0], &[1.0], &[0.0]]; // lets try XOR

    let mut network = Network::new(
        &[2, 16, 16, 1],
        &[
            Activation::LReLU(0.01),
            Activation::LReLU(0.01),
            Activation::Sigmoid,
        ],
    )?;
    let optimizer = Optimizer::new(0.01, 2, Loss::CrossEntropy);
    optimizer.train(&mut network, &inputs, &expected, EPOCHS)?;

    for input in inputs {
        println!("{:?} -> {:?}", input, network.predict(input));
    }

    Ok(())
}
