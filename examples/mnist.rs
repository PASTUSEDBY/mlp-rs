use anyhow::bail;
use image::{ColorType, ImageReader, Luma, imageops::FilterType};
use rand::random_range;
use std::{
    fs::{File, create_dir},
    io::{self, BufRead},
    path::Path,
};

use binrw::io::BufReader;
use mlp::{
    ModelError, Network, Optimizer,
    strategy::{Activation, Loss},
};

const MODEL: &'static str = "./cache/mnist_model.bin";
const MNIST_TRAIN: &'static str = "./cache/mnist_train.csv";
const MNIST_TEST: &'static str = "./cache/mnist_test.csv";

#[derive(Debug)]
struct TrainData {
    output: f64,
    data: Vec<f64>,
}

fn load_dataset(path: &str, num: usize, upper_limit: usize) -> anyhow::Result<Vec<TrainData>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let random_line = random_range(1..=upper_limit);
    println!(
        "Choosing line {} out of {}, will grab {} elements.",
        random_line,
        upper_limit + num,
        num
    );

    // lets skip the lines until our target
    // we use range inclusive to skip the first line
    let mut discard = String::new();
    for _ in 0..=random_line {
        discard.clear();
        reader.read_line(&mut discard)?;
    }

    let mut train_data = Vec::with_capacity(num);
    // lets train on the remaining lines
    for line in reader.lines().take(num) {
        let line = line?;
        let elems: Vec<f64> = line
            .split(",")
            .map(|e| e.trim().parse().expect("integer expected btw"))
            .collect();
        if elems.len() != 785 {
            bail!(
                "expected exactly 785 elements per line, comma separated. found {}",
                elems.len()
            );
        }

        // encode the output
        let output = elems[0];
        // and lets scale the data
        let data: Vec<f64> = elems.iter().skip(1).map(|e| e / 255.0).collect();

        train_data.push(TrainData { output, data });
    }

    Ok(train_data)
}

fn main() -> anyhow::Result<()> {
    let cache = Path::new("./cache");
    if !cache.exists() {
        create_dir(cache)?;
    }

    let res = File::open(MODEL)
        .map_err(ModelError::from)
        .and_then(|mut f| Network::load(&mut f));

    let mut network = match res {
        Ok(n) => n,
        Err(err) => {
            println!(
                "Model doesn't exist or is corrupted. Reason: {}\nCreating a new model...",
                err
            );
            Network::new(
                &[784, 64, 32, 10],
                &[
                    Activation::LReLU(0.01),
                    Activation::LReLU(0.01),
                    Activation::SoftMax,
                ],
            )?
        }
    };

    println!(
        "Enter if you wanna [train [epochs = 10]] the model, [test] or [predict <image_path>]: "
    );
    let mut line = String::new();
    io::stdin().read_line(&mut line)?;
    let (option, rest) = line
        .split_once(char::is_whitespace)
        .unwrap_or((line.as_ref(), ""));
    let option = option.to_ascii_lowercase();

    match option.as_ref() {
        "train" => {
            println!("Loading full data set, this might take a while to train...");
            let data = load_dataset(MNIST_TRAIN, 60_000, 1)?;
            let inputs: Vec<&[f64]> = data.iter().map(|t| t.data.as_slice()).collect();
            let expected_vectors: Vec<Vec<f64>> = data
                .iter()
                .map(|t| {
                    let mut out = vec![0.0; 10];
                    out[t.output as usize] = 1.0;
                    out
                })
                .collect();
            let expected: Vec<&[f64]> = expected_vectors.iter().map(|v| v.as_slice()).collect();

            let epochs = rest.trim().parse::<usize>().unwrap_or(10);
            println!("Will run for {epochs} epochs!");
            let opt = Optimizer::new(0.04, 128, Loss::CrossEntropy);
            opt.train_print_epoch(&mut network, &inputs, &expected, epochs, Some(true))?;
            println!("Successfully trained. Now we save!");
        }
        "test" => {
            const TEST: usize = 10000;
            let data = load_dataset(MNIST_TEST, TEST, 1)?;
            let inputs: Vec<&[f64]> = data.iter().map(|t| t.data.as_slice()).collect();
            let expected: Vec<f64> = data.iter().map(|t| t.output).collect();

            let mut success = 0.0;
            for (&input, exp) in inputs.iter().zip(expected) {
                let predicted = network.predict(input);
                let max = predicted
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap();
                if max == exp as usize {
                    success += 1.0;
                }
            }

            println!("Prediction rate: {}%", success / (TEST as f64) * 100.0);
        }
        "predict" => {
            let rest = rest.trim();
            if rest.is_empty() {
                bail!("where's the image path at for prediction?");
            }

            let image = ImageReader::open(rest)?
                .decode()?
                .resize_exact(28, 28, FilterType::Triangle)
                .into_luma8();

            let image_data = image
                .pixels()
                .map(|&Luma([val])| val as f64 / 255.0)
                .collect::<Vec<f64>>();

            image::save_buffer("./cache/seen.png", image.as_raw(), 28, 28, ColorType::L8)?;
            debug_assert_eq!(image_data.len(), 784usize);

            let predicted = network.predict(&image_data);
            let max = predicted
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();

            println!("Predicted: {max}");
            println!("More statistics (probability wise)");
            predicted
                .iter()
                .enumerate()
                .for_each(|(num, prob)| println!("{num}: {}%", prob * 100.0));
        }
        _ => bail!("nope, wrong option."),
    };

    // finally save it
    let mut file = File::options().write(true).create(true).open(MODEL)?;
    network.save(&mut file)?;

    Ok(())
}
