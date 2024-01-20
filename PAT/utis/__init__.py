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

import typing as t
from abc import ABC, abstractmethod
from collections import Counter

from ordered_set import OrderedSet
from xlsxwriter.worksheet import Worksheet


class NameAndDescription(ABC):
    @classmethod
    def name(cls):
        return cls.__name__

    @classmethod
    @abstractmethod
    def description(cls):
        raise NotImplementedError()


def create_table(
        data: t.Dict[str, t.Dict[str, str]], print_heading: bool = False, worksheet: Worksheet = None
) -> t.List[str]:
    all_keys = OrderedSet()
    for d in data.values():
        all_keys.update([str(x) for x in d.keys()])
    col_width = {
        x: max(len(str(y.get(x, ""))) for y in data.values()) for x in all_keys
    }
    if print_heading:
        col_width = {k: max(v, len(str(k))) for k, v in col_width.items()}
    col_width[None] = max(len(x) for x in data.keys())
    h = list(x for x in col_width.keys() if x is not None)

    ret = []
    if print_heading:
        ret.append(f" {' ' * col_width[None]} ║ {' │ '.join(f'{str(x):<{col_width[x]}}' for x in h)}")
        ret.append(f"═{'═' * col_width[None]}═╬═{'═╪═'.join('═' * col_width[x] for x in h)}")

    for row_key, row_value in data.items():
        ret.append(
            f" {row_key:<{col_width[None]}} ║ {' │ '.join(f'{row_value.get(x, str()):<{col_width[x]}}' for x in h)}")

    if worksheet is not None:
        row = 0
        if print_heading:
            for col, head in enumerate(h, 1):
                worksheet.write(row, col, str(head))
            row += 1

        for row, (row_key, row_value) in enumerate(data.items(), row):
            worksheet.write(row, 0, row_key)
            for col, head in enumerate(h, 1):
                if head in row_value:
                    worksheet.write(row, col, row_value[head])

    return ret


def counter_union(*counters: Counter) -> Counter:
    from collections import Counter
    ret = Counter()

    for c in counters:
        ret.update(c)

    return ret


def super_format(value, format_strs: t.Dict[t.Type, str]) -> str:
    for typ, fmt in format_strs.items():
        if isinstance(value, typ):
            return f"{value:{fmt}}"
    return ""


del ABC, abstractmethod, t, Counter, Worksheet

from .cliconfig import CLIConfig

__all__ = ['NameAndDescription', 'CLIConfig', "create_table"]
