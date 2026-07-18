# MLP-RS

A feed-forward Multi-Layer Perceptron implementation written from scratch in Rust.

The core library does not depend on external machine learning or linear algebra crates at all. The examples use a small number of utility crates such as `image` and `anyhow`.

The goal of this project is to try building one from scratch, of course.<br>
Which means I had to learn the Math from scratch too, especially how **backpropagation** heavily leans into calculus which I already like.

## Using the Network

```rust
let mut network = Network::new(
    &[784, 64, 32, 10],
    &[
        Activation::LReLU(0.01),
        Activation::LReLU(0.01),
        Activation::SoftMax,
    ],
)?;

// Optimize it
let opt = Optimizer::new(0.01, 32, Loss::CrossEntropy, ExecutionStrategy::Concurrent { workers: Some(4) });
opt.train(&mut network, &inputs, &expected, num_epochs)?;

// Finally save it maybe
network.save(&mut file);
```

## Features
- Mini Batch SGD Training
- Multiple Activation strategies, including:
    - Linear
    - Sigmoid
    - LeakyReLU _(use 0.0 to get vanilla ReLU)_
    - SoftMax
- Multiple Loss functions
    - Mean Squared Error
    - Cross Entropy (only works with `Sigmoid` or `Softmax`)
- Arbitrary number of layers initialization, specifying neurons and Activation strategy per layer
- Saving/loading the Network to a binary format

**Some examples are specified in the `examples`, you can try running one by `cargo run --release --example <file>`. The `main.rs` has the XOR example copied.**

## Examples
- [XOR](examples/xor.rs): The usual, learns the XOR gate.
- [MNIST](examples/mnist.rs): Classifying a set of handwritten digit dataset.
- [Function Approximation](examples/func.rs): Only approximates `sin(x)` from `-10π to 10π`.

> The [`trained_models`](trained_models/) directory contains the binary models for `Function Approximation` and `MNIST` for quicker access. Specifically, for now, the `MNIST` model has an accuracy of `97.36%` over the official data set of `10,000` images after about 1,100 epochs on the training set.
