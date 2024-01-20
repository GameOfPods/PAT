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

import json
import logging
from typing import Dict, Any, Optional, Tuple, Callable, Union, List

from PAT.modules import PATModule


class SpeakerDetectionModule(PATModule):
    _LOGGER = logging.getLogger("SpeakerDetectionModule")
    _PIPELINE = None

    def __init__(self, file: str):
        super().__init__(file)
        if self.__class__._PIPELINE is None:
            raise ValueError(f"{self.__class__.name()} did not get a load call to prepare pipeline")

    @classmethod
    def supports_file(cls, file: str) -> bool:
        try:
            import torchaudio
            _ = torchaudio.load(file)
            return True
        except (ImportError, KeyError, RuntimeError) as e:
            logging.error(e)
            return False

    @classmethod
    def description(cls) -> str:
        return "Module that analyzes speaker information from podcast files. Accepts files loadable by torchaudio"

    def process(self) -> Union[Tuple[Dict[str, Any], List[Tuple[str, bytes]]], Dict[str, Any]]:
        import os
        from io import StringIO
        from time import perf_counter
        from datetime import timedelta
        from collections import defaultdict
        import torchaudio
        from pyannote.audio.pipelines.utils.hook import ProgressHook
        from pyannote.core import Annotation, Segment
        from pydub import AudioSegment

        waveform, sample_rate = torchaudio.load(self.file)
        duration_seconds = waveform.shape[-1] / sample_rate
        self._LOGGER.info(f"Audio is {timedelta(seconds=duration_seconds)} long. Samplerate is {sample_rate}")

        with ProgressHook() as hook:
            data = {"waveform": waveform, "sample_rate": sample_rate}
            num_speakers = int(os.environ["SPEAKER_COUNT"]) if "SPEAKER_COUNT" in os.environ else None
            self._LOGGER.info(f"Running pipeline on {self.file} for {num_speakers} speakers")
            t1 = perf_counter()
            diarization: Annotation = self.__class__._PIPELINE(data, hook=hook, num_speakers=num_speakers)
            t2 = perf_counter()
            self._LOGGER.info(f"Diarization took {timedelta(seconds=t2 - t1)}")

        song = AudioSegment.from_file(self.file)
        speaker_segments = defaultdict(lambda: [])

        for segment, segment_id, label in diarization.itertracks(yield_label=True):
            segment: Segment
            speaker_segments[label].append((segment.start, segment.end))
            if False:
                song[segment.start * 1000:segment.end * 1000].export(
                    os.path.join("G:\\tmp", f"{label}-{len(speaker_segments[label])}.mp3")
                )

        speaker_counts = {k: len(v) for k, v in speaker_segments.items()}
        speaker_durations = {k: sum(v) for k, v in
                             ((k2, [v2e - v2s for v2s, v2e in v2]) for k2, v2 in speaker_segments.items())}
        self._LOGGER.info(
            f"Speaker presence: {', '.join(str(x) for x in sorted(speaker_counts.items(), key=lambda x: x[::-1], reverse=True))}"
        )
        self._LOGGER.info(
            f"Speaker duration: {', '.join(str(x) for x in sorted(speaker_durations.items(), key=lambda x: x[::-1], reverse=True))}"
        )

        rttm_file = StringIO()
        diarization.write_rttm(rttm_file)
        rttm_file.seek(0)

        return {
            "duration": duration_seconds,
            "sample_rate": sample_rate,
            "speaker_counts": speaker_counts,
            "speaker_durations": speaker_durations,
        }, [("speaker.rttm", rttm_file.read().encode("utf-8"))]

    @classmethod
    def load(cls, config: Dict[str, Any]):
        super().load(config=config)
        model_path: Optional[str] = config.get("model_path", None)
        optimized_data: Optional[str] = config.get("optimized_data", None)
        if cls._PIPELINE is None:
            import torch
            from pyannote.audio import Pipeline
            import os

            checkpoint = os.environ.get("PYANNOTE_PIPELINE", "pyannote/speaker-diarization-3.1").strip()
            auth_token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN", None)

            cls._LOGGER.info(f"Loading Pipeline {checkpoint} {'without' if auth_token is None else 'with'} auth token")
            cls._PIPELINE = Pipeline.from_pretrained(checkpoint_path=checkpoint, use_auth_token=auth_token)
            if model_path is not None:
                from pyannote.audio.pipelines import SpeakerDiarization
                cls._LOGGER.info(f"Will construct pipeline using \"{model_path}\" as segmentation model")
                cls._PIPELINE = SpeakerDiarization(
                    segmentation=model_path,
                    embedding=cls._PIPELINE.embedding,
                    embedding_exclude_overlap=cls._PIPELINE.embedding_exclude_overlap,
                    clustering=cls._PIPELINE.klustering,
                )

                if optimized_data is not None:
                    cls._LOGGER.info(f"Will instantiate pipeline using optimized data from \"{model_path}\"")
                    with open(optimized_data, "r") as f:
                        cls._PIPELINE = cls._PIPELINE.instantiate(json.load(f))

            if torch.cuda.is_available():
                dev = torch.device("cuda")
            else:
                dev = torch.device("cpu")
            cls._PIPELINE = cls._PIPELINE.to(dev)
            cls._LOGGER.info(f"Pipeline loaded and moved to {cls._PIPELINE.device}")

    @classmethod
    def config_keys(cls) -> Dict[str, Tuple[Callable[[str], Any], bool, str, Union[None, str, int]]]:
        return {
            "model_path": (str, False, "Set path to custom trained segmentation model", None),
            "optimized_data": (str, False, "Set path to optimized data to instantiate pipeline", None)
        }

    @classmethod
    def unload(cls):
        super().unload()
        if cls._PIPELINE is not None:
            try:
                cls._PIPELINE.unregister()
            except AttributeError:
                pass
            del cls._PIPELINE
