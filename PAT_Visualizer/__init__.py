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

__author__ = 'RedRem95'

# noinspection PyUnresolvedReferences
from PAT_Visualizer.modules import *

# noinspection DuplicatedCode
_sub_mods = [x.name() for x in VisualizerModule.__subclasses__()]
_one_doubled = False

for _i, _mod in enumerate(_sub_mods):
    if _mod in _sub_mods[:_i]:
        print(f"Found more than one module with name '{_mod}'")
        _one_doubled = True

if _one_doubled:
    exit(1)

del _i, _mod, _sub_mods, _one_doubled
