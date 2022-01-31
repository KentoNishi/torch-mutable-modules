import torch
import torch.nn as nn
from torch_mutable_modules import convert_to_mutable_module, mutable_module


######################################
# convert_to_mutable_module function #
######################################

# create a mutable linear layer
mutable_linear = convert_to_mutable_module(nn.Linear(1, 1))

# parameters are initially random
mutable_linear.weight  # tensor([[-0.7162]], grad_fn=<CloneBackward>)
mutable_linear.bias  # tensor([-0.8702], grad_fn=<CloneBackward>)

# do in-place operations on the weights
mutable_linear.weight.fill_(420)
# outright replace the bias tensor
mutable_linear.bias = torch.ones_like(mutable_linear.bias, requires_grad=True) * 0.69
mutable_linear.weight  # tensor([[420.]], grad_fn=<FillBackward0>)
mutable_linear.bias  # tensor([0.6900], grad_fn=<MulBackward0>)

# mutable layers can be used like normal modules
mutable_linear(torch.ones(1, 1))  # tensor([[420.6900]], grad_fn=<AddmmBackward>)


############################
# mutable_module decorator #
############################

# you can also use the mutable_module decorator to make a class mutable
@mutable_module  # declare the module class mutable
class MutableCustom(nn.Module):  # the module class
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)
        self.sequential = nn.Sequential(
            nn.Linear(1, 1),
            nn.Linear(1, 1),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.sequential(x)
        return x


# all parameters are recursively made mutable
mutable_custom_module = MutableCustom()
print(mutable_custom_module.conv.weight)
mutable_custom_module.conv.weight += 1