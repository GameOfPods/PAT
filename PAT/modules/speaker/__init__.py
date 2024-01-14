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
import logging

from PAT.modules import Module


class SpeakerDetectionModule(Module):
    _LOGGER = logging.getLogger("SpeakerDetectionModule")
    _PIPELINE = None

    def __init__(self, file: str):
        super().__init__(file)
        if self.__class__._PIPELINE is None:
            self.load()

    @classmethod
    def supports_file(cls, file: str) -> bool:
        try:
            import torchaudio
            _ = torchaudio.load(file)
            return True
        except (ImportError, KeyError) as e:
            logging.error(e)
            return False

    @classmethod
    def description(cls) -> str:
        return "Module that analyzes speaker information from podcast files. Accepts files loadable by torchaudio"

    def process(self):
        import os
        from time import perf_counter
        from datetime import timedelta
        from collections import defaultdict
        import torchaudio
        from pyannote.audio.pipelines.utils.hook import ProgressHook
        from pyannote.core import Annotation, Segment
        from pydub import AudioSegment

        waveform, sample_rate = torchaudio.load(self.file)

        with ProgressHook() as hook:
            data = {"waveform": waveform, "sample_rate": sample_rate}
            num_speakers = int(os.environ["SPEAKER_COUNT"]) if "SPEAKER_COUNT" in os.environ else None
            self._LOGGER.debug(f"Running pipeline on {self.file} for {num_speakers} speakers")
            t1 = perf_counter()
            diarization: Annotation = self.__class__._PIPELINE(data, hook=hook, num_speakers=num_speakers)
            t2 = perf_counter()
            self._LOGGER.info(f"Diarization took {timedelta(seconds=t2 - t1)}")

        song = AudioSegment.from_file(self.file)
        speaker_counts = defaultdict(lambda: 0)

        for segment, segment_id, label in diarization.itertracks(yield_label=True):
            segment: Segment
            speaker_counts[label] += 1
            song[segment.start * 1000:segment.end * 1000].export(
                os.path.join("G:\\tmp", f"{label}-{speaker_counts[label]}.mp3")
            )

        self._LOGGER.debug(
            f"Speaker presence: {', '.join(str(x) for x in sorted(speaker_counts.items(), key=lambda x: x[::-1], reverse=True))}")

        exit(1)

    @classmethod
    def load(cls):
        super().load()
        if cls._PIPELINE is None:
            import torch
            from pyannote.audio import Pipeline
            import os
            checkpoint = os.environ.get("PYANNOTE_PIPELINE", "pyannote/speaker-diarization-3.1").strip()
            auth_token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN", None)
            cls._LOGGER.info(f"Loading Pipeline {checkpoint} {'without' if auth_token is None else 'with'} auth token")
            cls._PIPELINE = Pipeline.from_pretrained(checkpoint_path=checkpoint, use_auth_token=auth_token)
            if torch.cuda.is_available():
                dev = torch.device("cuda")
            else:
                dev = torch.device("cpu")
            cls._PIPELINE.to(dev)
            cls._LOGGER.info(f"Pipeline loaded and moved to {dev}")

    @classmethod
    def unload(cls):
        super().unload()
        if cls._PIPELINE is not None:
            cls._PIPELINE.unregister()
            del cls._PIPELINE
