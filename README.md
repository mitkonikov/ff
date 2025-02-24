# PyTorch-based library for the Forward-Forward algorithm

FFLib is a neural network library based on PyTorch that aims to implement
several different types of layers and networks based on the Forward-Forward algorithm.
The library also provides a suite of tools for training, validating, testing, debugging and experimenting
with Forward-Forward-based networks. We aim to make this library as close as possible
to the original design and structure of the PyTorch library.

## Getting Started

### Installing dependencies

PyTorch and torchvision are the only main dependencies needed to run all of the examples for now.

> We are working on establishing a `requirements.txt` file and a `conda` env.

### Examples

To get started with our library check out the examples in the `./examples` folder.
You can run the examples either from the CLI or from inside VSCode interactively.
We recommend running them from the CLI for the first time, so you can fix any dependency issues
and making interactive execution only afterwards.


## The Forward-Forward Algorithm

The Forward-Forward Algorithm was introduced in Geoffrey Hinton's paper
["The Forward-Forward Algorithm: Some Preliminary Investigations"](https://arxiv.org/abs/2212.13345)
with the following abstract:

```
The aim of this paper is to introduce a new learning procedure for neural networks and to demonstrate that it works well enough on a few small problems to be worth further investigation. The Forward-Forward algorithm replaces the forward and backward passes of backpropagation by two forward passes, one with positive (i.e. real) data and the other with negative data which could be generated by the network itself. Each layer has its own objective function which is simply to have high goodness for positive data and low goodness for negative data. The sum of the squared activities in a layer can be used as the goodness but there are many other possibilities, including minus the sum of the squared activities. If the positive and negative passes could be separated in time, the negative passes could be done offline, which would make the learning much simpler in the positive pass and allow video to be pipelined through the network without ever storing activities or stopping to propagate derivatives.
```

## Contributions

We really appreciate contributions from the community!
We especially welcome the reports of issues and bugs.

However, one may note that since this library is currently being heavily developed,
the API may drastically change and all projects depending on this library have to deal
with the changes downstream. We will however try to keep these at minimum.

The main maintainer of this library is [Mitko Nikov](https://github.com/mitkonikov).

## Guidelines

Here are a few guidelines to following while contributing on the library:
 - We aim to keep this library with as little run-time-necessary dependencies as possible.
 - Unit tests for as many functions as possible. (we know that we can't cover everything)
 - Strict Static Type-checking using `mypy`
 - Strict formatting style guidelines (to be configured)
 - No recursion (at our abstraction level)
 - Nicely documented functions and classes
