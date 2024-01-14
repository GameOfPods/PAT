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

import glob
import logging
import os.path
import sys
from argparse import ArgumentParser
from typing import List, Tuple, Type, Optional

from PAT import Module, __version__


def main():

    parser = ArgumentParser(prog="PAT",
                            description=f"PodcastProject Analytics Toolkit v{__version__}\n"
                                        f"PAT  Copyright (C) 2024  RedRem95 (GNU GENERAL PUBLIC LICENSE)")

    parser.add_argument("-V", "--version", action="version", version=f'%(prog)s {__version__}')
    parser.add_argument("-ls", dest="ls", action="store_true", help="list all loaded modules and exit")
    parser.add_argument("input", help="input files you want to process", nargs="*", default=list())
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", help="More verbose")

    args = parser.parse_args()

    if args.ls:
        sub_classes = [(x.name(), x.description()) for x in Module.__subclasses__()]
        print(f"Modules ({len(sub_classes)}):")
        for m_name, m_desc in sub_classes:
            print(f" - {m_name:{max(len(x[0]) for x in sub_classes)}s} | {m_desc}")
        exit(0)

    logging.basicConfig(
        format="{asctime} - {levelname:^8} - {name}: {message}",
        style="{",
        encoding='utf-8',
        datefmt="%Y.%m.%d %H:%M:%S",
        level=logging.DEBUG if args.verbose else logging.INFO,
        stream=sys.stdout,
    )

    tasks: List[Tuple[Type[Module], Tuple[str, ...]]] = []

    modules = Module.__subclasses__()

    in_files = []
    for in_file in args.input:
        in_files.extend((x for x in (os.path.abspath(y) for y in glob.glob(in_file)) if os.path.exists(x)))

    for i in range(len(in_files) - 1, 0, -1):
        if in_files[i] in in_files[0:i]:
            in_files.pop(i)

    print(f"Processing {len(in_files)} files")

    for in_file in sorted(in_files):
        in_file_path = os.path.abspath(in_file)
        in_file_name = os.path.basename(in_file)
        acc_module = [x for x in modules if x.supports_file(file=in_file)]
        print(f"\"{in_file_name}\": accepted by {len(acc_module)} modules ({', '.join(x.name() for x in acc_module)})")
        for m in acc_module:
            tasks.append((m, (in_file_path, )))

    tasks.sort(key=lambda x: (x[0].name(), x[0].__name__))

    print(f"Executing {len(tasks)} tasks")

    prev_module: Optional[Type[Module]] = None

    for i, (m, args) in enumerate(tasks):
        if prev_module != m:
            if prev_module is not None:
                prev_module.unload()
            m.load()
        print(f"{i+1:{len(str(len(tasks)))}}/{len(tasks)}: Executing task {m.name()} on \"{args[0]}\"")
        module_instance = m(*args)
        module_instance.process()
        prev_module = m


if __name__ == "__main__":
    main()
