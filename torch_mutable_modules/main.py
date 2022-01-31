import torch.nn as nn
from typing import TypeVar, Type

_T = TypeVar("_T", bound=nn.Module)


def _clone_param(param: nn.parameter.Parameter) -> nn.parameter.Parameter:
    return param.data.clone().detach().requires_grad_(True).clone()


def convert_to_mutable_module(module: _T) -> _T:
    """
    Convert a module to a mutable module.

    `module` (`torch.nn.Module`): The module to convert into a mutable module.
    """
    try:

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

        cached_parameters = (
            module.named_parameters() if hasattr(module, "named_parameters") else []
        )
        converted_module = MutableModule(module)
        for name, param in cached_parameters:
            split_name = name.split(".")
            if len(split_name) > 1:
                convert_to_mutable_module(param)
            else:
                object.__setattr__(
                    module,
                    name,
                    _clone_param(param),
                )
        return converted_module

    except:
        return module


# mutable_module decorator
def mutable_module(module_class: Type[_T]) -> Type[_T]:
    """
    Decorator for converting PyTorch modules into mutable modules.

    ```python
    @mutable_module
    class MyModule(nn.Module):
        pass
    ```
    """

    def constructor(*args, **kwargs) -> _T:
        return convert_to_mutable_module(module_class(*args, **kwargs))

    return constructor
