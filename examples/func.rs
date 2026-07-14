use binrw::io::BufReader;
use rand::random_range;
use std::{
    f64::consts::PI, fs::{File, create_dir}, io::{self, BufWriter, Seek, SeekFrom, Write}, path::Path,
};

use mlp::{
    ModelError, Network, Optimizer,
    strategy::{Activation, ExecutionStrategy, Loss},
};

const MODEL: &'static str = "./cache/func_model.bin";
const FACTOR: f64 = 10.0 * PI;

fn target(x: f64) -> f64 {
    x.sin()
}

fn train(
    network: &mut Network,
    factor: f64,
    batches: usize,
    epochs: usize,
    strategy: ExecutionStrategy,
) -> anyhow::Result<()> {
    let opt = Optimizer::new(0.01, 64, Loss::MeanSquare, strategy);
    let cap = 512 * batches;
    let mut inputs = Vec::with_capacity(cap);
    let mut outputs = Vec::with_capacity(cap);

    for _ in 0..cap {
        let x = random_range(-factor..=factor);
        inputs.push(vec![x / factor]);
        outputs.push(vec![target(x)]);
    }

    opt.train(network, inputs, outputs, epochs)?;

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let cache = Path::new("./cache");
    if !cache.exists() {
        create_dir(cache)?;
    }

    let res = File::open(MODEL)
        .map(BufReader::new)
        .map_err(ModelError::from)
        .and_then(|mut f| Network::load(&mut f));

    let mut network = match res {
        Ok(n) => n,
        Err(err) => {
            println!(
                "Model doesn't exist or is corrupted. Reason: {}\nCreating a new model...",
                err
            );
            let mut net = Network::new(
                &[1, 32, 16, 8, 1],
                &[
                    Activation::LReLU(0.01),
                    Activation::LReLU(0.01),
                    Activation::LReLU(0.01),
                    Activation::Linear,
                ],
            )?;
            println!("Mandatory first time training, please wait a bit...");
            train(
                &mut net,
                FACTOR,
                100,
                1_000,
                ExecutionStrategy::Sequential
            )?;
            net
        }
    };

    // train a bit every run
    train(&mut network, FACTOR, 100, 10, ExecutionStrategy::Sequential)?;
    let mut file = File::options()
        .write(true)
        .create(true)
        .open(MODEL)
        .map(BufWriter::new)?;

    loop {
        // save before every loop
        file.seek(SeekFrom::Start(0))?;
        file.get_mut().set_len(0)?;

        network.save(&mut file)?;
        file.flush()?;
        println!("Enter x (or q):");

        let mut line = String::new();
        io::stdin().read_line(&mut line)?;

        let line = line.trim();

        if line.eq_ignore_ascii_case("q") {
            println!("Bye");
            break;
        }

        let x: f64 = line.parse()?;

        let pred = network.predict(&[x / FACTOR])[0];

        println!(
            "actual = {:.6}, predicted = {:.6}, error = {:.6}",
            target(x),
            pred,
            (pred - target(x)).abs()
        );
    }

    Ok(())
}
