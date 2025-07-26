# NN-DL-implementation

## Overview

This repository provides a **minimal yet comprehensive implementation of a fully-connected neural network in NumPy**, designed for educational purposes and experimentation. The implementation covers essential deep learning concepts, including custom activation functions, forward and backward propagation, gradient descent, and model evaluation. 

The code is structured to be accessible for learners and serves as a strong foundation for extending to more complex architectures and tasks.

---

## Features

- **Manual Neural Network Construction**: No external deep learning libraries (e.g., TensorFlow, PyTorch) required—everything is built from scratch using NumPy.
- **Customizable Architecture**: Change number of input features, hidden units, and output classes easily.
- **Activation Functions**: Implements Sigmoid, ReLU, and Tanh, with their derivatives for backpropagation.
- **Forward and Backward Propagation**: Step-by-step computation of all intermediate values and gradients.
- **Cross-Entropy Loss**: Standard cost function for binary classification.
- **Gradient Descent Optimizer**: Updates weights and biases using computed gradients.
- **Training and Prediction Utilities**: Simple interface for training the model and making predictions.
- **Demonstration with Synthetic Data**: Example for binary classification with randomly generated data.

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Implementation Details](#implementation-details)
- [Sample Output](#sample-output)
- [Extending This Project](#extending-this-project)
- [References](#references)
- [License](#license)

---

## Installation

No installation required beyond Python 3 and NumPy.

```bash
pip install numpy
```

---

## Usage

Run the main script to train a neural network on a synthetic dataset:

```bash
python ShallowNN_From_Scratch.py
```

**Example output:**
```
Cost after iteration 0: 0.6932
Cost after iteration 1000: 0.6054
Cost after iteration 2000: 0.2114
Cost after iteration 3000: 0.1302
Cost after iteration 4000: 0.0985
Training accuracy: 99.80%
```

---

## Implementation Details

### 1. Activation Functions

- **Sigmoid**: `1 / (1 + exp(-z))`
- **ReLU**: `max(0, z)`
- **Tanh**: `np.tanh(z)`
- **Derivatives**: All derivatives are explicitly defined for use in backpropagation.

### 2. Parameter Initialization

Weights and biases are initialized with small random values (weights) and zeros (biases) for both layers, ensuring efficient convergence.

### 3. Forward Propagation

- Computes intermediate activations and outputs using current weights, biases, and activation functions.
- Stores necessary intermediate values for backpropagation.

### 4. Cost Computation

- Utilizes binary cross-entropy loss:
  
  \[
  L = - \frac{1}{m} \sum [y \log(a) + (1 - y) \log(1 - a)]
  \]

### 5. Backward Propagation

- Calculates gradients of loss with respect to all parameters using the chain rule.
- Gradients are used to update weights and biases.

### 6. Parameter Update

- Uses gradient descent to update parameters.

### 7. Model Training & Prediction

- `model(...)`: Trains the network for a specified number of iterations.
- `predict(...)`: Makes predictions on new data.

### 8. Demonstration

- Trains on a synthetic binary classification task: classifies points as above or below a linear boundary.

---

## Sample Output

```
Cost after iteration 0: 0.6932
Cost after iteration 1000: 0.6054
Cost after iteration 2000: 0.2114
Cost after iteration 3000: 0.1302
Cost after iteration 4000: 0.0985
Training accuracy: 99.80%
```

---

## Extending This Project

- **Add More Layers**: Implement support for multi-layer (deep) neural networks.
- **Different Activation Functions**: Try Leaky ReLU, Softmax, or others.
- **Regularization**: Add L2/L1 regularization to mitigate overfitting.
- **Mini-batch Gradient Descent**: Improve performance on large datasets.
- **Support for Multiclass Classification**: Replace sigmoid with softmax and generalize loss function.
- **Visualization**: Plot decision boundaries and training curves.

---

## References

- Deep Learning Specilisation - Neural Networks and Deep Learning (Course 1)(https://www.coursera.org/learn/neural-networks-deep-learning)

---

## License

This project is released under the MIT License.

---
