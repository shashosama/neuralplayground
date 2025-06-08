# Simple NumPy Neural Network

This project implements a minimal neural network using NumPy from scratch. It performs binary classification on a dataset `X` and `y` using a two-layer architecture with sigmoid activations and backpropagation.

## Features

- Two-layer neural network (input → hidden → output)
- Sigmoid activation function
- Manual backpropagation
- Binary classification
- Uses NumPy only – no external ML libraries

## Code Overview

```python
# Architecture:
# Input → Hidden Layer (4 neurons) → Output (1 neuron)

W1 = np.random.randn(2, 4)
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1)
b2 = np.zeros((1, 1))

