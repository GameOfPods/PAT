import os
from abc import ABC, abstractmethod, abstractclassmethod
from typing import Callable, Optional


class Module(ABC):
    def __init__(self, file: str):
        self._file = file

    @property
    def file(self) -> str:
        return self._file

    @classmethod
    def name(cls):
        return cls.__name__

    @classmethod
    @abstractmethod
    def description(cls):
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def supports_file(cls, file: str) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def process(self):
        pass


__all__ = ["Module"]
_my_path = os.path.dirname(__file__)
for f in os.listdir(_my_path):
    full_path = os.path.join(_my_path, f)
    if os.path.isdir(full_path) and os.path.exists(os.path.join(full_path, '__init__.py')):
        if f in __all__:
            print(f"Already found module {f}")
        __all__.append(f)
    elif os.path.isfile(full_path) and full_path.endswith(".py"):
        if f[:-3] in __all__:
            print(f"Already found module {f}")
        __all__.append(f[:-3])


