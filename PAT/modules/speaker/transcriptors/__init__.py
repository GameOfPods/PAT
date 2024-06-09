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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class WordTuple:
    start: Optional[float]
    end: Optional[float]
    word: Optional[str]


@dataclass
class WordTupleSpeaker:
    start: Optional[float]
    end: Optional[float]
    word: Optional[str]
    speaker: Optional[str]

    @classmethod
    def from_word_tuple(cls, word: WordTuple, speaker: Optional[str]) -> 'WordTupleSpeaker':
        return WordTupleSpeaker(start=word.start, end=word.end, word=word.word, speaker=speaker)


class Transcriptor(ABC):
    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def transcribe(
            cls, audio_file: str, model: str, language: Optional[str] = None, device: Optional[str] = None
    ) -> Tuple[List[WordTuple], Optional[str]]:
        raise NotImplementedError()


from PAT.modules.speaker.transcriptors.localwhisper import LocalWhisperTranscriptor

__all__ = ["WordTuple", "WordTupleSpeaker", "Transcriptor", "LocalWhisperTranscriptor"]
