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

import glob
import json
import logging
import os.path
import sys
from argparse import ArgumentParser
from datetime import datetime, timedelta
from pathlib import Path
from time import perf_counter
from typing import List, Tuple, Type, Optional

from PAT import PATModule, __version__
from PAT.utis.cliconfig import CLIConfig


def main():
    sub_classes = sorted((x.name(), x.description(), x) for x in PATModule.__subclasses__())

    parser = ArgumentParser(prog="PAT",
                            description=f"PodcastProject Analytics Toolkit v{__version__} \n"
                                        f"PAT  Copyright (C) 2024  RedRem95 (GNU GENERAL PUBLIC LICENSE) \n"
                                        f"{len(sub_classes)} Modules present (See -ls for details)")

    parser.add_argument("-V", "--version", action="version", version=f'%(prog)s {__version__}')
    parser.add_argument("-ls", dest="ls", action="store_true", help="list all loaded modules and exit")
    parser.add_argument("input", help="input files you want to process", nargs="*", default=list())
    parser.add_argument("-r", "--recursive", dest="recursive", action="store_true",
                        help="Search files recursive. Can be dangerous and lead to a huge amount of files")
    parser.add_argument("-t", "--target", dest="target", type=Path, required=False, default=os.getcwd(),
                        help="target directory. Will automatically add current day sub folder [default: %(default)s]")
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", help="More verbose")

    CLIConfig.add_config_to_parser(parser=parser, subclasses=[x[2] for x in sub_classes])

    args = parser.parse_args()

    if args.ls:
        print(f"Modules ({len(sub_classes)}):")
        for m_name, m_desc, *_ in sub_classes:
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

    logger = logging.getLogger("PAT")

    t_start = perf_counter()

    tasks: List[Tuple[Type[PATModule], Tuple[str, ...]]] = []

    in_files = []
    for in_file in args.input:
        # print(in_file, glob.glob(in_file), glob.glob(in_file, recursive=args.recursive))
        in_files.extend((x for x in (os.path.abspath(y) for y in glob.glob(in_file, recursive=args.recursive)) if
                         os.path.exists(x) or True))

    for i in range(len(in_files) - 1, 0, -1):
        if in_files[i] in in_files[0:i]:
            in_files.pop(i)

    target_folder = None
    while target_folder is None or target_folder.exists():
        target_folder = args.target.joinpath(f'A_{datetime.now().strftime("%Y%m%d-%H%M")}')
    os.makedirs(target_folder, exist_ok=False)
    logger.info(f"Saving results to {target_folder}")

    logger.info(f"Processing {len(in_files)} files")

    for in_file in sorted(in_files):
        in_file_path = os.path.abspath(in_file)
        in_file_name = os.path.basename(in_file)
        acc_module = [x[-1] for x in sub_classes if x[-1].supports_file(file=in_file)]
        logger.info(
            f"\"{in_file_name}\": accepted by {len(acc_module)} modules ({', '.join(x.name() for x in acc_module)})")
        for m in acc_module:
            tasks.append((m, (in_file_path,)))

    tasks.sort(key=lambda x: (x[0].name(), x[0].__name__))

    logger.info(f"Executing {len(tasks)} tasks")

    prev_module: Optional[Type[PATModule]] = None

    for i, (m, arguments) in enumerate(tasks):
        if prev_module != m:
            if prev_module is not None:
                prev_module.unload()
            conf = CLIConfig.parse_parser_data(m=m, args=args)
            m.load(config=conf)
        logger.info(f"{i + 1:{len(str(len(tasks)))}}/{len(tasks)}: Executing task {m.name()} on \"{arguments[0]}\"")
        t1 = perf_counter()
        module_instance = m(*arguments)
        ret = module_instance.process()
        t2 = perf_counter()
        logger.info(f"{m.name()} on \"{arguments[0]}\" took {timedelta(seconds=t2 - t1)}")
        if isinstance(ret, dict):
            infos, special_files = ret, []
        elif isinstance(ret, tuple) and len(ret) == 2 and isinstance(ret[0], dict) and isinstance(ret[1], list):
            infos, special_files = ret
        else:
            logger.error(f"Output of module {m.name()} on {arguments[0]} did not produce a savable result")
            continue

        out_name = f"{i}_{m.name()}_{Path(arguments[0]).stem}"
        special_files_path = target_folder.joinpath(out_name)
        i_dict = {
            "path": arguments[0],
            "module": m.name(),
            "module_class": str(m),
            "infos": infos,
        }

        if len(special_files) > 0:
            if special_files_path.exists():
                logger.error(f"Something went wrong with folder creation. {special_files_path} already exists")
            os.mkdir(special_files_path)
            for special_file_name, special_file_content in special_files:
                with open(special_files_path.joinpath(special_file_name), "wb") as f_special_file:
                    f_special_file.write(special_file_content)
            i_dict["additional_files"] = [os.path.join(special_files_path.name, x) for x, *_ in special_files]

        with open(target_folder.joinpath(f"{out_name}.json"), "w") as f_json:
            json.dump(i_dict, f_json, indent=2)

        prev_module = m

    if prev_module is not None:
        prev_module.unload()

    t_end = perf_counter()

    logger.info(f"Processing of {len(tasks)} took {timedelta(seconds=t_end - t_start)}")
    logger.info(f"Results saved to \"{target_folder}\"")


if __name__ == "__main__":
    main()
