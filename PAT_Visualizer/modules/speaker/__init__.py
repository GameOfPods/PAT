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
from collections import defaultdict

from PAT import PATModule
from PAT_Visualizer.modules import VisualizerModule


class SpeakerVisualizer(VisualizerModule):
    _SPEAKER_DATA: t.Dict[int, t.Dict[int, t.Dict[str, t.List[dict]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: [])))

    @classmethod
    def load(cls, config: t.Dict[str, t.Any]):
        super().load(config)
        cls._SPEAKER_DATA.clear()

    @classmethod
    def unload(cls):
        super().unload()
        cls._SPEAKER_DATA.clear()

    def __init__(self, data: t.Dict[str, t.Any]):
        super().__init__(data)

    @classmethod
    def supported_modules(cls) -> t.List[t.Type[PATModule]]:
        from PAT.modules.speaker import SpeakerDetectionModule
        return [SpeakerDetectionModule, ]

    def visualize(self) -> t.Generator[str, str, t.Optional[t.Dict[str, bytes]]]:
        from pathlib import Path
        import re

        data = self.data
        file_path = Path(data["path"]).absolute()
        infos = data["infos"]
        speaker, season, episode = None, None, None
        walk_path = file_path
        while speaker is None or episode is None or season is None:

            if (season is None or episode is None) and re.fullmatch(r"\d+\.\d+", walk_path.name):
                season, episode = int(walk_path.name.split(".")[0]), int(walk_path.name.split(".")[1])
            if speaker is None and walk_path.name.lower() in ("alex", "max"):
                speaker = walk_path.name.capitalize()

            if walk_path == walk_path.parent:
                raise Exception()
            walk_path = walk_path.parent

        yield f"Recording by {speaker} for S{season}E{episode}"

        d = {
            "season": season,
            "episode": episode,
            "speaker": speaker,
            "Duration": infos["duration"],
            "Sample rate": infos["sample_rate"],
            "Speaker count": sum(infos["speaker_counts"].values()),
            "Speaker duration": sum(infos["speaker_durations"].values())
        }

        self._SPEAKER_DATA[season][episode][speaker].append(d)

        return

    @classmethod
    def visualize_final(cls) -> t.Generator[str, str, t.Optional[t.Dict[str, bytes]]]:
        if len(cls._SPEAKER_DATA) > 0:
            from io import BytesIO
            from datetime import timedelta

            import xlsxwriter
            from PAT.utis import create_table

            def mean(_i):
                if len(_i) >= 1:
                    return sum(_i) / len(_i)
                return 0

            tbl_file = BytesIO()
            workbook = xlsxwriter.Workbook(tbl_file, options={"strings_to_numbers": True})

            tbl: t.Dict[str, t.Dict[str, t.Any]] = {}

            for season in sorted(cls._SPEAKER_DATA.keys()):
                for episode in sorted(cls._SPEAKER_DATA[season].keys()):
                    for speaker in sorted(cls._SPEAKER_DATA[season][episode].keys()):
                        d = cls._SPEAKER_DATA[season][episode][speaker].copy()
                        if len(d) <= 0:
                            continue
                        data = {
                            "season": season,
                            "episode": episode,
                            "speaker": speaker,
                        }
                        data.update({
                            k: sum(x.get(k, 0) for x in d) for k in ["Duration", "Speaker count", "Speaker duration"]
                        })
                        data["Part speaking"] = data["Speaker duration"] / data["Duration"]
                        data["Average speaking length"] = data["Speaker duration"] / data["Speaker count"]
                        data["Sample rate"] = mean([x["Sample rate"] for x in d if "Sample rate" in x])
                        data["Duration str"] = str(timedelta(seconds=data["Duration"]))

                        tbl[f"S{season}E{episode} - {speaker}"] = data

            for l in create_table(print_heading=True, data=tbl, worksheet=workbook.add_worksheet("DATA")):
                yield l
            workbook.close()
            return {"tables.xlsx": tbl_file.getbuffer().tobytes()}
        return
