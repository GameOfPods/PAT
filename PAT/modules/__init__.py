#  PAT - Toolkit to analyze podcast audio and topics talked about in the podcast. For example Books
#  Copyright (c) 2024.  RedRem95
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.

import os
from abc import ABC, abstractmethod


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

    @classmethod
    def load(cls):
        print(f"First run of {cls.name()}")
        pass

    @classmethod
    def unload(cls):
        print(f"No run of {cls.name()} anymore")
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

del os, ABC, abstractmethod
