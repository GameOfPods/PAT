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

from typing import Dict, List, Tuple


def main(
        rttm_file: str, target_folder: str,
        snippet_length: int = 5, exp_format: str = "mp3", cut: bool = False,
        database_name: str = "PodcastDatabase", protocol_name: str = "Protocol",
):
    import os
    from collections import defaultdict
    from tqdm import tqdm
    import random
    from datetime import timedelta
    from functools import lru_cache
    from math import ceil

    import yaml
    from pydub import AudioSegment
    from sklearn.model_selection import train_test_split

    from modules.speaker.diarizators.pyannote.prepare_rttm import RTTM_LINE

    @lru_cache
    def get_audio_segment(_full_path: str) -> AudioSegment:
        return AudioSegment.from_file(_full_path)

    rttm_file = os.path.abspath(rttm_file)
    base_folder = os.path.dirname(rttm_file)
    available_files = [x for x in (os.path.join(base_folder, y) for y in os.listdir(base_folder)) if os.path.isfile(x)]
    speaker_snippets: Dict[str, List[Tuple[str, float, float]]] = defaultdict(lambda: list())

    with open(rttm_file, 'r') as f:
        for line in f.readlines():
            _, audio_file, _, start, dur, _, _, speaker_name, _, _ = [x for x in line.strip().split(" ") if len(x) > 0]
            start, dur = float(start), float(dur)
            valid_files = [x for x in available_files if os.path.basename(x).startswith(f"{audio_file}.")]
            if len(valid_files) <= 0:
                continue
            elif len(valid_files) > 1:
                print(f"More than one file matching {audio_file}")
                continue
            full_path = valid_files[0]
            snippets = [(full_path, start, dur)]
            while cut and any(x[2] > snippet_length / 3 for x in snippets):
                new_snippets = []
                for snippet in snippets:
                    if snippet[2] < snippet_length / 3:
                        new_snippets.append(snippet)
                    else:
                        new_snippets.append((snippet[0], snippet[1], snippet[2] / 2))
                        new_snippets.append((snippet[0], snippet[1] + snippet[2] / 2, snippet[2] / 2))
                snippets = new_snippets
            speaker_snippets[speaker_name].extend(snippets)

    print(f"Found {len(speaker_snippets)} different speakers: {', '.join(speaker_snippets.keys())}")
    print(f"      {sum(len(x) for x in speaker_snippets.values())} unique snippets of speech")
    print(f"      {len(set().union(*[[y[0] for y in x] for x in speaker_snippets.values()]))} unique input audio")
    for speaker_name, snippets in speaker_snippets.items():
        print(
            f"{speaker_name:{max(len(x) for x in speaker_snippets.keys())}} has {len(snippets)} snippets with a duration of {timedelta(seconds=sum(x[2] for x in snippets))}")

    annotations = []
    uem = []
    snippet_files = []

    with tqdm(desc=f"Create training audio snippets of length {snippet_length}s", leave=True, unit="f") as pb:
        while any(len(x) > 0 for x in speaker_snippets.values()):
            file_name = f"audio_{len(snippet_files)}"
            seg = AudioSegment.silent(duration=snippet_length * 1000)

            speaker_cutoffs = {k: 0 for k in speaker_snippets.keys()}

            while True:
                speakers = [k for k, v in speaker_snippets.items() if len(v) > 0]
                if len(speakers) == 0 or max(speaker_cutoffs.values()) >= (snippet_length * 1000) - 500:
                    break
                summ = sum(len(speaker_snippets[k]) for k in speakers)

                speaker_prob = [[k, len(speaker_snippets[k])] for k in speakers]
                for i in range(len(speaker_prob)):
                    speaker_prob[i][1] += sum(x[1] for x in speaker_prob[:i])

                speaker_selection = random.randint(0, summ)
                try:
                    sel_speaker = [x for x, y in speaker_prob if speaker_selection <= y][0]
                except IndexError as e:
                    print(speaker_prob, summ, speaker_selection)
                    raise e
                selected_snippet_id = random.randint(0, len(speaker_snippets[sel_speaker]) - 1)
                selected_snippet = speaker_snippets[sel_speaker].pop(selected_snippet_id)
                full_path, start, dur = selected_snippet

                snippet = get_audio_segment(_full_path=full_path)[start * 1000:(start + dur) * 1000]

                snippet_start = max(
                    speaker_cutoffs[sel_speaker],
                    max(speaker_cutoffs.values()) + random.randint(-snippet_length * 100, snippet_length * 100)
                )
                snippet_end = min(snippet_start + ceil(dur * 1000), snippet_length * 1000)
                snippet_dur = snippet_end - snippet_start
                seg = seg.overlay(seg=snippet[0:snippet_dur], position=snippet_start, loop=False)
                # snippet.export(os.path.join(target_folder, f"{uuid4()}.wav"), format="wav")
                speaker_cutoffs[sel_speaker] = snippet_end + 10
                annotations.append(
                    {"file": file_name, "start": snippet_start / 1000, "dur": snippet_end / 1000, "name": sel_speaker}
                )

            uem.append({"file": file_name, "start": 0, "end": snippet_length})
            seg.export(os.path.join(target_folder, f"{file_name}.{exp_format}"), format=exp_format)
            snippet_files.append(file_name)

            pb.update()

    print(f"Created {len(snippet_files)} audio files to train with. In these are {len(annotations)} snippets of speech")

    with open(os.path.join(target_folder, "speaker.rttm"), "w") as rttm:
        rttm.write("\n".join(RTTM_LINE.format(**x) for x in annotations))

    with open(os.path.join(target_folder, "uem.uem"), "w") as uem_file:
        uem_file.write("\n".join("{file} NA {start:.f} {end:.f}".format(**x) for x in uem))

    train_snippets, _ = train_test_split(snippet_files, train_size=0.7)
    test_snippets, dev_snippets = train_test_split(_, train_size=2 / 3)

    with open(os.path.join(target_folder, "train.lst"), "w") as train_file:
        train_file.write("\n".join(train_snippets))
    with open(os.path.join(target_folder, "test.lst"), "w") as test_file:
        test_file.write("\n".join(test_snippets))
    with open(os.path.join(target_folder, "dev.lst"), "w") as test_file:
        test_file.write("\n".join(dev_snippets))

    with open(os.path.join(target_folder, "database.yml"), "w") as database_file:
        database_file.write(
            yaml.dump(
                {"Protocols": {database_name: {"SpeakerDiarization": {protocol_name: {
                    "scope": "file",
                    "train": {"uri": "train.lst", "annotation": "speaker.rttm", "annotated": "uem.uem"},
                    "test": {"uri": "test.lst", "annotation": "speaker.rttm", "annotated": "uem.uem"},
                    "development": {"uri": "dev.lst", "annotation": "speaker.rttm", "annotated": "uem.uem"}
                }}}}},
                default_flow_style=False
            ))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Prepare the training data")
    parser.add_argument('-f', required=True, dest="rttm_file", help="rttm file")
    parser.add_argument('-t', required=True, dest="target", help="Target folder")
    parser.add_argument("-l", required=False, default=5, type=int, dest="length", help="length of audio snippets")
    parser.add_argument("-c", required=False, action="store_true", dest="cut", help="cut longer parts")
    parser.add_argument("--database", default="PodcastDatabase", dest="database",
                        help="Name of the Database to be created for training in the resulting database.yml file")
    parser.add_argument("--protocol", default="Protocol", dest="protocol",
                        help="Name of the Protocol to be created for training in the resulting database.yml file")
    parser.add_argument()

    args = parser.parse_args()

    main(
        rttm_file=args.rttm_file, snippet_length=args.length, target_folder=args.target, exp_format="wav", cut=args.cut,
        database_name=args.database, protocol_name=args.protocol
    )
