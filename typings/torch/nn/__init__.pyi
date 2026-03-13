from collections.abc import Iterable

from torch import Tensor

class Module:
    training: bool

    def __init__(self) -> None: ...
    def __call__(self, *args: object, **kwargs: object) -> Tensor: ...
    def eval(self) -> Module: ...
    def load_state_dict(self, state_dict: dict[str, object]) -> None: ...

class ModuleList(Module):
    def __init__(self, modules: Iterable[Module]) -> None: ...

class Sequential(Module):
    def __init__(self, *modules: Module) -> None: ...

class Conv1d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        dilation: int = ...,
        padding: int = ...,
    ) -> None: ...

class BatchNorm1d(Module):
    def __init__(self, num_features: int) -> None: ...

class ReLU(Module):
    def __init__(self) -> None: ...

class Dropout(Module):
    def __init__(self, p: float = ...) -> None: ...

class Identity(Module):
    def __init__(self) -> None: ...

class Linear(Module):
    def __init__(self, in_features: int, out_features: int) -> None: ...
