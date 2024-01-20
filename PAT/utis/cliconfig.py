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

import typing as t
from abc import ABC
from argparse import ArgumentParser, Namespace

from . import NameAndDescription


class CLIConfig(NameAndDescription, ABC):
    @classmethod
    def config_keys(cls) -> t.Dict[str, t.Tuple[t.Callable[[str], t.Any], bool, str, t.Union[None, str, int]]]:
        return dict()

    @classmethod
    def load(cls, config: t.Dict[str, t.Any]):
        from logging import getLogger
        getLogger(cls.__name__).info(f"First run of {cls.name()}")
        pass

    @classmethod
    def unload(cls):
        from logging import getLogger
        getLogger(cls.__name__).info(f"No run of {cls.name()} anymore")
        pass

    @classmethod
    def add_config_to_parser(cls, parser: ArgumentParser, subclasses: t.Iterable[t.Type["CLIConfig"]]):
        for sc_class in subclasses:
            sc_name = sc_class.name()
            for arg, (tpe, required, hlp, nargs) in sc_class.config_keys().items():
                arg_name = f"{sc_name}__{arg}"
                parser.add_argument(
                    f"--{arg_name}", dest=arg_name, type=tpe, metavar=arg.upper(),
                    help=hlp, required=required, nargs=nargs,
                )

    @classmethod
    def parse_parser_data(cls, m: t.Type["CLIConfig"], args: Namespace):
        return {k[len(m.name()) + 2:]: v for k, v in args.__dict__.items() if k.startswith(m.name())}
