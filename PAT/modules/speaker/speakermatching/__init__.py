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
from logging import getLogger
from os import PathLike
from typing import Tuple, Union, BinaryIO, Dict, Optional


class SpeakerMatching:
    _LOGGER = getLogger(__name__)

    @classmethod
    def is_available(cls) -> bool:
        try:
            import librosa.feature
            from sklearn.mixture import GaussianMixture
            return True
        except ImportError:
            return False

    def __init__(self, train_audio: Dict[str, Union[str, PathLike, BinaryIO]]):
        import librosa.feature
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.neighbors import KNeighborsClassifier

        self._name = {}
        train_x, train_y = [], []
        for i, (name, data) in enumerate(train_audio.items()):
            try:
                self._LOGGER.debug(f"Loading file {data} for {name}")
                y, sr = self._load_audio(audio=data)
                self._LOGGER.debug(f"Calculating features for {name}")
                features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
                train_x.extend([z for z in features.transpose()])
                train_y.extend([i for _ in range(features.shape[1])])
                self._name[i] = name
                self._LOGGER.info(f"Created mfcc features for {name}")
            except Exception as e:
                self._LOGGER.error(f"Failed to load features for {name}: {e}")

        self._model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
        self._model.fit(train_x, train_y)
        self._LOGGER.info(f"Trained classifier to distinguish {len(self._name)} different speakers")

    def match_speaker(self, audio: Union[str, PathLike, BinaryIO]) -> Optional[Tuple[str, float]]:
        try:
            from collections import Counter
            import librosa.feature
            import numpy as np
            y, sr = self._load_audio(audio=audio)
            features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            pred: np.ndarray = self._model.predict(features.transpose())
            counter = Counter(pred)
            return self._name[counter.most_common(n=1)[0][0]], counter.most_common(n=1)[0][1] / counter.total()
        except Exception as e:
            self._LOGGER.error(f"Failed to predict speaker: {e}")

    @classmethod
    def _load_audio(cls, audio: Union[str, PathLike, BinaryIO], sample_rate: Optional[int] = 16000):
        import librosa.effects
        import numpy as np
        from time import perf_counter
        from datetime import timedelta as td

        t1 = perf_counter()
        y, sr = librosa.load(audio, sr=sample_rate, mono=True)
        t2 = perf_counter()
        cls._LOGGER.info(
            f"Load audio in {td(seconds=t2 - t1)} for speaker matching with length of {td(seconds=y.shape[0] / sr)}"
        )

        try:
            from silero_vad import load_silero_vad, get_speech_timestamps
            model = load_silero_vad()
            t1 = perf_counter()
            speech_timestamps = get_speech_timestamps(y, model, sampling_rate=sr)
            t2 = perf_counter()
            cls._LOGGER.info(f"Voice Activity Detection using silerio_vad took {td(seconds=t2 - t1)}")
        except ImportError:
            intervals = librosa.effects.split(y)
            speech_timestamps = [{"start": i, "end": j} for i, j in intervals]
            cls._LOGGER.info(f"Voice Activity Detection using librosa took {td(seconds=t2 - t1)}")

        if len(speech_timestamps) > 0:
            cls._LOGGER.info(f"Found {len(speech_timestamps)} intervals in audio. ")
            t1 = perf_counter()
            r = []
            for interval in speech_timestamps:
                r.extend(list(range(interval["start"], interval["end"])))
            y2 = y[np.array(r)]
            t2 = perf_counter()
            cls._LOGGER.info(f"Merging took {td(seconds=t2 - t1)}. New length is {td(seconds=y2.shape[0] / sr)}")

        else:
            y2 = y
        # sf.write("test.mp3", y2.reshape(-1, 1), int(sr))
        return y2, sr
