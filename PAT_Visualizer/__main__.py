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

from typing import Optional


def main():
    import os
    import sys
    import logging
    import glob
    from argparse import ArgumentParser
    import json
    from pathlib import Path
    from datetime import datetime

    from PAT_Visualizer import VisualizerModule, __version__
    from PAT.utis.cliconfig import CLIConfig

    # noinspection DuplicatedCode
    sub_classes = sorted((x.name(), x.description(), x) for x in VisualizerModule.__subclasses__())

    parser = ArgumentParser(prog="PAT-Visualizer",
                            description=f"Visualizer for the PodcastProject Analytics Toolkit v{__version__} \n"
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

    # noinspection DuplicatedCode
    target_folder: Optional[Path] = None
    while target_folder is None or target_folder.exists():
        target_folder: Path = args.target.joinpath(f'V_{datetime.now().strftime("%Y%m%d-%H%M")}')
    os.makedirs(target_folder, exist_ok=False)
    logger.info(f"Saving results to {target_folder}")

    in_files = []
    for in_file in args.input:
        # print(in_file, glob.glob(in_file), glob.glob(in_file, recursive=args.recursive))
        in_files.extend((x for x in (os.path.abspath(y) for y in glob.glob(in_file, recursive=args.recursive)) if
                         os.path.isfile(x) or True))

    mod_conf = {module: CLIConfig.parse_parser_data(m=module, args=args) for _, _, module in sub_classes}

    logger.info(f"Processing {len(in_files)} files")

    with open(target_folder.joinpath("log.txt"), "w", encoding="utf-8") as f_log:

        def p_log(a):
            f_log.write(f"{a}\n")
            print(a)

        for module_name, module_description, module in sub_classes:
            module.load(config=mod_conf[module])

        for i, in_file in enumerate(in_files):
            in_file = Path(in_file)
            with open(in_file, "r") as f_in:
                data = json.load(f_in)
            for module_name, module_description, module in sub_classes:
                data = data.copy()
                if module.supports_data(data=data):
                    p_log(f"╭─ {i + 1:{len(str(len(in_files)))}}/{len(in_files)} '{in_file.name}' by '{module_name}':")
                    p_log(f"│")
                    viz_class = module(data=data)
                    _iter = viz_class.visualize()
                    while True:
                        try:
                            line = next(_iter)
                            p_log(f"│ {line}")
                        except StopIteration as e:
                            if e.value is not None and isinstance(e.value, dict):
                                mod_target_folder = target_folder.joinpath(f"{i}_{module_name}_{in_file.name}")
                                os.makedirs(mod_target_folder, exist_ok=False)
                                for f_name, f_data in e.value.items():
                                    with open(mod_target_folder.joinpath(f_name), "wb") as f_out:
                                        f_out.write(f_data)
                            break
                    p_log(f"│")
                    p_log(f"╰─ {module_name}")

        for i, (module_name, module_description, module) in enumerate(sub_classes):
            p_log(f"╭─ {i + 1:{len(str(len(sub_classes)))}}/{len(sub_classes)} '{module_name}'-Module:")
            p_log(f"│")
            _iter = module.visualize_final()
            while True:
                try:
                    line = next(_iter)
                    p_log(f"│ {line}")
                except StopIteration as e:
                    if e.value is not None and isinstance(e.value, dict):
                        mod_target_folder = target_folder.joinpath(f"{i}_{module_name}")
                        os.makedirs(mod_target_folder, exist_ok=False)
                        for f_name, f_data in e.value.items():
                            with open(mod_target_folder.joinpath(f_name), "wb") as f_out:
                                f_out.write(f_data)
                    break
            p_log(f"│")
            p_log(f"╰─ {module_name}")
            module.unload()


if __name__ == "__main__":
    main()
