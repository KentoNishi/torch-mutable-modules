# Example usage of the torch_mutable_modules package

from typing import List
import torch
import torch.nn as nn
from torch_mutable_modules import convert_to_mutable_module, mutable_module

torch.manual_seed(1984)


def test_convert_to_mutable_module_function():
    """
    Use the `convert_to_mutable_module` function to convert a PyTorch module to a mutable module.
    """

    # create a mutable linear layer
    mutable_linear = convert_to_mutable_module(nn.Linear(1, 1))

    # parameters are initially random
    assert torch.is_tensor(mutable_linear.weight)
    assert torch.is_tensor(mutable_linear.bias)

    # do in-place operations on the weights
    mutable_linear.weight.fill_(420)
    # outright replace the bias tensor
    mutable_linear.bias = (
        torch.ones_like(mutable_linear.bias, requires_grad=True) * 0.69
    )
    assert str(mutable_linear.weight) == "tensor([[420.]], grad_fn=<FillBackward0>)"
    assert str(mutable_linear.bias) == "tensor([0.6900], grad_fn=<MulBackward0>)"

    # mutable layers can be used like normal modules
    assert (
        str(mutable_linear(torch.ones(1, 1)))
        == "tensor([[420.6900]], grad_fn=<AddmmBackward>)"
    )


def test_mutable_module_decorator():
    """
    Use the `mutable_module` decorator to convert a PyTorch module into a mutable module.
    """

    @mutable_module  # declare the module class mutable
    class MutableCustom(nn.Module):  # the module class
        def __init__(self):
            super().__init__()
            self.convs: List[nn.Conv2d] = nn.ModuleList(
                [nn.Conv2d(1, 1, 1), nn.Conv2d(1, 1, 1)]
            )

        def forward(self, x):
            x = self.convs[0](x)
            x = self.convs[1](x)
            return x

    # all parameters are recursively made mutable
    mutable_custom_module = MutableCustom()
    mutable_custom_module.convs[0].weight *= 0
    mutable_custom_module.convs[0].weight += 420
    assert (
        str(mutable_custom_module.convs[0].weight)
        == "tensor([[[[420.]]]], grad_fn=<AddBackward0>)"
    )
