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
from typing import Set

RTTM_LINE = "SPEAKER {file} 1 {start} {dur} <NA> <NA> {name} <NA> <NA>"


def prepare_rttm(files: Set[str], target_folder):
    import os
    from collections import defaultdict
    from pathlib import Path

    from tqdm import tqdm
    import torch
    import torchaudio
    from pyannote.audio import Pipeline
    from pyannote.core import Annotation, Segment

    pipeline = Pipeline.from_pretrained(checkpoint_path="pyannote/speaker-diarization-3.1", use_auth_token=None)
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    pipeline.to(dev)

    print(f"Preparing in {len(files)} files")
    name_counter = defaultdict(lambda: 0)

    new_annotation = []

    for file in tqdm(sorted(files), desc="Iterating files", unit="f"):
        file = os.path.abspath(file)
        name = os.path.basename(os.path.dirname(file)).capitalize()
        name_counter[name] += 1

        new_filename = os.path.join(target_folder, f"{name}-{name_counter[name]}.mp3")
        waveform, sample_rate = torchaudio.load(file)

        torchaudio.save(new_filename, waveform, sample_rate)

        diarization: Annotation = pipeline({"waveform": waveform, "sample_rate": sample_rate}, num_speakers=1)

        for segment, track_id, *_ in diarization.itertracks(yield_label=True):
            segment: Segment
            new_annotation.append(
                {"file": Path(new_filename).stem, "start": segment.start, "dur": segment.duration, "name": name}
            )

    rttm_file = os.path.join(target_folder, f"speaker.rttm")
    with open(rttm_file, "w") as rttm:
        rttm.write("\n".join(RTTM_LINE.format(**x) for x in new_annotation))


if __name__ == '__main__':
    from argparse import ArgumentParser
    import glob
    import os

    arg_parser = ArgumentParser()
    arg_parser.add_argument('INPUTS', nargs="+", help="Input files")
    arg_parser.add_argument("-t", dest="target", required=True, help="Target directory")

    args = arg_parser.parse_args()

    file_collection = set()
    for f in args.INPUTS:
        file_collection.update(os.path.abspath(x) for x in glob.glob(f))

    prepare_rttm(files=file_collection, target_folder=args.target)
