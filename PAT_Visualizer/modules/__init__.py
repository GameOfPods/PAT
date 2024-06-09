#  PAT - Toolkit to analyze podcast audio and topics talked about in the podcast. For example Books
#  Copyright (c) 2024.  RedRem95
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.

import os
import typing as t
from abc import ABC, abstractmethod

from PAT import PATModule
from PAT.utils import NameAndDescription
from PAT.utils.cliconfig import CLIConfig


class VisualizerModule(CLIConfig, NameAndDescription, ABC):
    def __init__(self, data: t.Dict[str, t.Any]):
        self._data = data

    @classmethod
    def load(cls, config: t.Dict[str, t.Any]):
        return

    @classmethod
    def unload(cls):
        return

    @property
    def data(self) -> t.Dict[str, t.Any]:
        return self._data.copy()

    @classmethod
    @abstractmethod
    def supported_modules(cls) -> t.List[t.Type[PATModule]]:
        raise NotImplementedError()

    @classmethod
    def description(cls):
        return (f"Visualizer for results produced by {'either ' if len(cls.supported_modules()) > 1 else ''}"
                f"{' or '.join(x.name() for x in cls.supported_modules())}")

    @classmethod
    def supports_data(cls, data: t.Dict[str, t.Any]) -> bool:
        try:
            return any(str(x) == data["module_class"] and x.name() == data["module"] for x in cls.supported_modules())
        except KeyError:
            pass
        return False

    @abstractmethod
    def visualize(self) -> t.Generator[str, str, t.Optional[t.Dict[str, bytes]]]:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def visualize_final(cls) -> t.Generator[str, str, t.Optional[t.Dict[str, bytes]]]:
        raise NotImplementedError()


__all__ = ['VisualizerModule']
# noinspection DuplicatedCode
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

del os, ABC, abstractmethod, t, _my_path, full_path
