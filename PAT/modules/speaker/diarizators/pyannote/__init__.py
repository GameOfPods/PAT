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


import logging
import os
from datetime import timedelta
from time import perf_counter
from typing import List, Optional

from rttm_manager.entity.rttm import RTTM

from PAT.modules.speaker.diarizators import Diarizators


class PyannoteDiarizator(Diarizators):
    _LOGGER = logging.getLogger("PyannoteDiarizator")

    SEGMENTATION_MODEL: Optional[str] = None
    OPTIMIZED_DATA: Optional[str] = None

    @classmethod
    def is_available(cls) -> bool:
        try:
            import rttm_manager
            import torchaudio
            import pyannote.audio
            import pyannote.core
        except ImportError:
            return False
        return True

    @classmethod
    def diarize(cls, audio_file: str, temp_dir: str, device: str = None) -> List[RTTM]:
        import torchaudio
        from pyannote.audio.pipelines.utils.hook import ProgressHook
        from pyannote.core import Annotation
        from rttm_manager import import_rttm

        waveform, sample_rate = torchaudio.load(audio_file)

        duration_seconds = waveform.shape[-1] / sample_rate
        cls._LOGGER.info(f"Audio is {timedelta(seconds=duration_seconds)} long. Samplerate is {sample_rate}")

        with ProgressHook() as hook:
            data = {"waveform": waveform, "sample_rate": sample_rate}
            num_speakers = int(os.environ["SPEAKER_COUNT"]) if "SPEAKER_COUNT" in os.environ else None

            cls._LOGGER.info(f"Creating diarization pipeline")
            t1 = perf_counter()
            pipeline = cls._create_pipeline()
            t2 = perf_counter()
            cls._LOGGER.info(f"Pipeline creating took {timedelta(seconds=t2 - t1)}")

            cls._LOGGER.info(f"Running pipeline on {audio_file} for {num_speakers} speakers")
            t1 = perf_counter()
            diarization: Annotation = pipeline(data, hook=hook, num_speakers=num_speakers)
            t2 = perf_counter()
            cls._LOGGER.info(f"Diarization took {timedelta(seconds=t2 - t1)}")

        with open(os.path.join(temp_dir, "diarization.rttm"), "w") as rttm_f:
            diarization.write_rttm(rttm_f)

        return import_rttm(os.path.join(temp_dir, "diarization.rttm"))

    @classmethod
    def _create_pipeline(cls, device: Optional[str] = None):
        import json
        import torch
        from pyannote.audio import Pipeline

        checkpoint = os.environ.get("PYANNOTE_PIPELINE", "pyannote/speaker-diarization-3.1").strip()
        auth_token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN", None)
        model_path = cls.SEGMENTATION_MODEL
        optimized_data = cls.OPTIMIZED_DATA

        cls._LOGGER.info(f"Loading Pipeline {checkpoint} {'without' if auth_token is None else 'with'} auth token")
        pipeline = Pipeline.from_pretrained(checkpoint_path=checkpoint, use_auth_token=auth_token)
        if model_path is not None:
            from pyannote.audio.pipelines import SpeakerDiarization
            cls._LOGGER.info(f"Will construct pipeline using \"{model_path}\" as segmentation model")
            pipeline = SpeakerDiarization(
                segmentation=model_path,
                embedding=pipeline.embedding,
                embedding_exclude_overlap=pipeline.embedding_exclude_overlap,
                clustering=pipeline.klustering,
            )

            if optimized_data is not None:
                cls._LOGGER.info(f"Will instantiate pipeline using optimized data from \"{optimized_data}\"")
                with open(optimized_data, "r") as f:
                    pipeline = pipeline.instantiate(json.load(f))

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        return pipeline.to(device)
