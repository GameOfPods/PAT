import random
from typing import Callable, Optional

from . import Module
from multiprocessing import current_process


class SpeakerModule(Module):
    @classmethod
    def supports_file(cls, file: str) -> bool:
        return file.endswith(".mp3")

    @classmethod
    def description(cls):
        return "Module that analyzes speakers in provided audio files"

    def process(self):
        pass


