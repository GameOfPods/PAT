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
from typing import List, Tuple
from typing import Optional

from PAT.modules.speaker.transcriptors import Transcriptor, WordTuple


class LocalWhisperTranscriptor(Transcriptor):
    _LOGGER = logging.getLogger(__name__)

    _DEFAULT_TYPES = {"cpu": "int8", "cuda": "int8"}

    @classmethod
    def is_available(cls) -> bool:
        try:
            import torch
            import faster_whisper
        except ImportError:
            return False
        return True

    @classmethod
    def transcribe(
            cls, audio_file: str, model: str, language: Optional[str] = None, device: Optional[str] = None
    ) -> Tuple[List[WordTuple], Optional[str]]:
        import torch
        import tqdm
        from faster_whisper import WhisperModel
        from time import perf_counter
        from datetime import timedelta

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        cls._LOGGER.info(f"Transcribing audio using Whisper and \"{model}\" model. Selected language: {language}")
        model = WhisperModel(
            model, device=device, compute_type=os.environ.get("WHISPER_COMPUTE_TYPE", cls._DEFAULT_TYPES[device])
        )

        segments, info = model.transcribe(
            audio_file,
            language=language,
            beam_size=5,
            vad_filter=True,
        )

        segments_dictionary = []
        t1 = perf_counter()
        for segment in tqdm.tqdm(segments, unit="segment", leave=False, desc="Transcribing"):
            segments_dictionary.append(segment._asdict())
        t2 = perf_counter()
        cls._LOGGER.info(f"Transcribed audio using Whisper. "
                         f"Found {len(segments_dictionary)} segments in audio, took {timedelta(seconds=t2 - t1)}")

        del model
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        word_timestamps: List[WordTuple] = []
        try:
            import whisperx
            from whisperx.alignment import DEFAULT_ALIGN_MODELS_HF, DEFAULT_ALIGN_MODELS_TORCH
            if info.language in set().union(
                    DEFAULT_ALIGN_MODELS_TORCH.keys(), DEFAULT_ALIGN_MODELS_HF.keys()
            ):
                cls._LOGGER.info(f"Aligning words using WhisperX")
                align_model, meta = whisperx.load_align_model(language_code=info.language, device=device)
                t1 = perf_counter()
                aligned = whisperx.align(
                    transcript=segments_dictionary,
                    model=align_model,
                    align_model_metadata=meta,
                    audio=audio_file,
                    device=device,
                    print_progress=False,
                )
                word_timestamps = _fix_broken_times(
                    words=[
                        WordTuple(
                            start=x.get("start", None), end=x.get("end", None), word=x.get("word", None)
                        ) for x in aligned["word_segments"]
                    ],
                    init=segments_dictionary[0].get("start", None),
                    fin=segments_dictionary[-1].get("end", None),
                )
                t2 = perf_counter()
                cls._LOGGER.info(f"Aligned words using WhisperX. "
                                 f"Found {len(word_timestamps)} timestamped words, took {timedelta(seconds=t2 - t1)}")
                del align_model
                if device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                cls._LOGGER.error(f"Failed to align whisper results. Language {info.language} not supported")
            del whisperx
            del DEFAULT_ALIGN_MODELS_HF, DEFAULT_ALIGN_MODELS_TORCH
        except ImportError:
            cls._LOGGER.error("WhisperX not installed. Failed to align")

        if word_timestamps is None or len(word_timestamps) == 0:
            cls._LOGGER.error("Failed to align words using WhisperX. Doing naive alignment")
            word_timestamps = []
            for s in segments_dictionary:
                for w in s["words"]:
                    word_timestamps.append(WordTuple(start=w[0], end=w[1], word=w[2]))

        return word_timestamps, info.language


def _merge_words(words: List[WordTuple], fin: float, idx: int = 0) -> Optional[float]:
    '''
    Merge words to previous that don't have a start timestamp
    Args:
        word_timestamps:
        current_word_index:
        final_timestamp:

    Returns:

    '''
    # if current word is the last word
    if idx >= len(words) - 1:
        return words[-1].start
    n = idx + 1
    while idx < len(words) - 1:
        if words[n].start is None:
            words[idx].word += f" {words[n].word}" if words[n].word else ""
            words[n].word = None
            if words[n].end is not None:
                return words[n].end
            if n + 1 >= len(words):
                return fin
            n += 1

        else:
            return words[n].start


def _fix_broken_times(
        words: List[WordTuple], init: Optional[float] = 0, fin: Optional[float] = None
) -> List[WordTuple]:
    if len(words) == 0:
        return words
    if words[0].start is None:
        words[0].start = init if init is not None else 0
    if words[0].end is None:
        words[0].end = _merge_words(words=words, idx=0, fin=fin)

    res = [words[0]]

    for i, w in enumerate(words[1:], start=1):
        if w.word is None:
            continue
        if w.start is None:
            w.start = words[i - 1].end
        if w.end is None:
            w.end = _merge_words(words=words, idx=i, fin=fin)
        res.append(w)
    return res
