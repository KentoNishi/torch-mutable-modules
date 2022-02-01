# Torch Mutable Modules

Use in-place and assignment operations on PyTorch module parameters with support for autograd.

[![Publish to PyPI](https://github.com/KentoNishi/torch-mutable-modules/actions/workflows/publish.yaml/badge.svg)](https://github.com/KentoNishi/torch-mutable-modules/actions/workflows/publish.yaml)
[![Run tests](https://github.com/KentoNishi/torch-mutable-modules/actions/workflows/test.yaml/badge.svg)](https://github.com/KentoNishi/torch-mutable-modules/actions/workflows/test.yaml)
[![PyPI version](https://img.shields.io/pypi/v/torch-mutable-modules.svg?style=flat)](https://pypi.org/project/torch-mutable-modules/)
[![Number of downloads from PyPI per month](https://img.shields.io/pypi/dm/torch-mutable-modules.svg?style=flat)](https://pypi.org/project/torch-mutable-modules/)
![Python version support](https://img.shields.io/pypi/pyversions/torch-mutable-modules)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/ambv/black)

## Why does this exist?

PyTorch does not allow in-place operations on module parameters (usually desirable):

```python
linear_layer = torch.nn.Linear(1, 1)
linear_layer.weight.data += 69
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Valid, but will NOT store grad_fn=<AddBackward0>
linear_layer.weight += 420
# ^^^^^^^^^^^^^^^^^^^^^^^^
# RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
```

In some cases, however, it is useful to be able to modify module parameters in-place. For example, if we have a neural network (`net_1`) that predicts the parameter values to another neural network (`net_2`), we need to be able to modify the weights of `net_2` in-place and backpropagate the gradients to `net_1`.

```python
# create a parameter predictor network (net_1)
net_1 = torch.nn.Linear(1, 2)

# predict the weights and biases of net_2 using net_1
p_weight_and_bias = net_1(input_0).unsqueeze(2)
p_weight, p_bias = p_weight_and_bias[:, 0], p_weight_and_bias[:, 1]

# create a mutable network (net_2)
net_2 = to_mutable_module(torch.nn.Linear(1, 1))

# hot-swap the weights and biases of net_2 with the predicted values
net_2.weight = p_weight
net_2.bias = p_bias

# compute the output and backpropagate the gradients to net_1
output = net_2(input_1)
loss = criterion(output, label)
loss.backward()
optimizer.step()
```

This library provides a way to easily convert PyTorch modules into mutable modules with the `to_mutable_module` function.

## Installation
You can install `torch-mutable-modules` from [PyPI](https://pypi.org/project/torch-mutable-modules/).

```bash
pip install torch-mutable-modules
```

To upgrade an existing installation of `torch-mutable-modules`, use the following command:

```bash
pip install --upgrade --no-cache-dir torch-mutable-modules
```

## Importing

You can use wildcard imports or import specific functions directly:

```python
# import all functions
from torch_mutable_modules import *

# ... or import the function manually
from torch_mutable_modules import to_mutable_module
```

## Usage

To convert an existing PyTorch module into a mutable module, use the `to_mutable_module` function:

```python
converted_module = to_mutable_module(
    torch.nn.Linear(1, 1)
) # type of converted_module is still torch.nn.Linear

converted_module.weight *= 0
convreted_module.weight += 69
convreted_module.weight # tensor([[69.]], grad_fn=<AddBackward0>)
```

You can also declare your own PyTorch module classes as mutable, and all child modules will be recursively converted into mutable modules:

```python
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

my_module = to_mutable_module(MyModule())
my_module.linear.weight *= 0
my_module.linear.weight += 69
my_module.linear.weight # tensor([[69.]], grad_fn=<AddBackward0>)
```

### ⚠ Warning: `.cuda()` and similar methods are not supported as of yet. Please replace individual parameters with CUDA tensors for the time being.

### Detailed examples

Please check out [example.py](./example.py) to see more detailed example usages of the `to_mutable_module` function.

## Contributing
Please feel free to submit issues or pull requests!
