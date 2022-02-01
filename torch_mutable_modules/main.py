import torch.nn as nn
from typing import TypeVar, Type

_T = TypeVar("_T", bound=nn.Module)


def _clone_param(param: nn.parameter.Parameter) -> nn.parameter.Parameter:
    return param.data.clone().detach().requires_grad_(True).clone()


def to_mutable_module(module: _T) -> _T:
    """
    Convert a module to a mutable module.

    `module` (`torch.nn.Module`): The module to convert into a mutable module.
    """

    class MutableModule(*module.__class__.mro()):
        def __init__(self, module: _T):
            object.__setattr__(self, "_module", module)
            super(nn.Module, self).__init__()

        def __getattribute__(self, name: str):
            module = object.__getattribute__(self, "_module")
            if name == "_module":
                return module
            return getattr(module, name)

        def __setattr__(self, name: str, value):
            module = object.__getattribute__(self, "_module")
            object.__setattr__(module, name, value)

        def __repr__(self):
            return f"MutableModule({self._module})"

    converted_module = MutableModule(module)

    for name, param in list(module.named_parameters()):
        split_name = name.split(".")
        if len(split_name) > 1:
            if isinstance(param, nn.Module):
                to_mutable_module(param)
            else:
                parent_object = converted_module
                base_object = getattr(converted_module, split_name[0])
                for part in split_name[1:-1]:
                    parent_object = base_object
                    base_object = getattr(base_object, part)
                setattr(
                    parent_object,
                    split_name[-2],
                    to_mutable_module(base_object),
                )

        else:
            object.__setattr__(
                module, name, _clone_param(param),
            )
    return converted_module

