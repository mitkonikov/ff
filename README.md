# Forward-Forward Neural Networks Library based on PyTorch

FFLib is a neural network library based on PyTorch that aims to implement
several different types of layers and networks based on the Forward-Forward algorithm.
The library also provides a suite of tools for training, validating, testing, debugging and experimenting
with Forward-Forward-based networks. We aim to make this library as close as possible
to the original design and structure of the PyTorch library.

<img src="https://raw.githubusercontent.com/mitkonikov/ff/refs/heads/main/docs/figures/logo.png" width="40%">

## Getting Started

### Installing dependencies

Dependencies needed to run the code and/or the examples:
 - [PyTorch](https://pytorch.org/)
 - [torchvision](https://pytorch.org/)
 - [tqdm](https://tqdm.github.io/)

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
The aim of this paper is to introduce a new learning procedure for neural networks and to
demonstrate that it works well enough on a few small problems to be worth further investigation.
The Forward-Forward algorithm replaces the forward and backward passes of backpropagation by
two forward passes, one with positive (i.e. real) data and the other with negative data which
could be generated by the network itself. Each layer has its own objective function which is
simply to have high goodness for positive data and low goodness for negative data.
The sum of the squared activities in a layer can be used as the goodness
but there are many other possibilities, including minus the sum of the squared activities.
If the positive and negative passes could be separated in time,
the negative passes could be done offline, which would make the learning much simpler
in the positive pass and allow video to be pipelined through the network without ever
storing activities or stopping to propagate derivatives.
```

## Contributions

We really appreciate contributions from the community!
We especially welcome the reports of issues and bugs.

However, one may note that since this library is currently being heavily developed,
the API may drastically change and all projects depending on this library have to deal
with the changes downstream. We will however try to keep these at minimum.

The main maintainer of this library is [Mitko Nikov](https://github.com/mitkonikov).

### Developing the library

We are using [poetry](https://python-poetry.org/) to manage, build and publish the python package.
We recommend downloading poetry and running `poetry install` to
install all of the dependencies instead of doing so manually.

To activate the virtual env created by poetry, run `poetry env activate` to get the
command to activate the env. After activation, you can run anything from within.

### Contributing to GitHub

There are three things that we are very strict about:
 - Type-checking - powered by [mypy](https://mypy-lang.org/)
 - Coding style - powered by [Black](https://black.readthedocs.io/en/stable/)
 - Unit Tests - powered by [pytest](https://docs.pytest.org/en/stable/)

Run the following commands in the virtual env
to ensure that everything is according to the guidelines:

```sh
mypy . --strict
black .
pytest .
```

Guidelines are now checked using GitHub Workflows.
When developing the library locally, you can install [act](https://nektosact.com/) to run
the GitHub workflows on your machine through Docker.
We also recommend installing the VSCode extension
[GitHub Local Actions](https://marketplace.visualstudio.com/items?itemName=SanjulaGanepola.github-local-actions)
to run the workflows from inside VSCode, making the process painless.

## General Guidelines

Here are a few guidelines to following while contributing on the library:
 - We aim to keep this library with as little run-time-necessary dependencies as possible.
 - Unit tests for as many functions as possible. (we know that we can't cover everything)
 - Strict Static Type-checking using `mypy`
 - Strict formatting style guidelines using `black`
 - No recursion (at our abstraction level)
 - Nicely documented functions and classes
