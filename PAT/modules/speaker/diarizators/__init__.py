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
from typing import List

from rttm_manager import RTTM


class Diarizators(ABC):

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def diarize(cls, audio_file: str, temp_dir: str, device: str = None) -> List[RTTM]:
        raise NotImplementedError()


from PAT.modules.speaker.diarizators.nemo import NemoDiarizator
from PAT.modules.speaker.diarizators.pyannote import PyannoteDiarizator

__all__ = ['Diarizators', 'NemoDiarizator', 'PyannoteDiarizator']
