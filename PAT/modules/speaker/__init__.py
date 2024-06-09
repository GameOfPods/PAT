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
from typing import Dict, Any, Optional, Tuple, Union, List, Type

from tqdm import tqdm

from PAT.modules import PATModule
from PAT.modules.speaker.diarizators import Diarizators, PyannoteDiarizator
from PAT.modules.speaker.transcriptors import Transcriptor, WordTupleSpeaker
from PAT.utils.punctuation import restore_punctuation


class SpeakerDetectionModule(PATModule):
    _LOGGER = logging.getLogger("SpeakerDetectionModule")
    _PIPELINE = None
    _TEMP_DIR = os.path.join(os.getcwd(), "temp_data")
    _DEMUCS_MODEL = "htdemucs"

    _AVAILABLE_DIARIZERS = {x.__name__: x for x in Diarizators.__subclasses__() if x.is_available()}
    if len(_AVAILABLE_DIARIZERS) == 0:
        _LOGGER.error("No diarizators available")
    else:
        _LOGGER.info(f"{len(_AVAILABLE_DIARIZERS)} diarizators available. {', '.join(_AVAILABLE_DIARIZERS.keys())}")
    _DIARIZER: Optional[Type[Diarizators]] = None

    _AVAILABLE_TRANSCRIPTORS = {x.__name__: x for x in Transcriptor.__subclasses__() if x.is_available()}
    if len(_AVAILABLE_TRANSCRIPTORS) == 0:
        _LOGGER.error("No transcriptors available")
    else:
        _LOGGER.info(
            f"{len(_AVAILABLE_DIARIZERS)} transcriptors available: {', '.join(_AVAILABLE_TRANSCRIPTORS.keys())}")
    _TRANSCRIPTOR: Optional[Type[Transcriptor]] = None
    _TRANSCRIPTOR_MODEL: Optional[str] = None
    _TRANSCRIPTOR_LANGUAGE: Optional[str] = None
    _SUMMARIZE_MODEL: Optional[str] = None

    def __init__(self, file: str):
        super().__init__(file)

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
        import torchaudio

        waveform, sample_rate = torchaudio.load(self.file)
        duration_seconds = waveform.shape[-1] / sample_rate
        self._LOGGER.info(f"Audio has duration of {timedelta(seconds=int(duration_seconds))}@{sample_rate}")
        del waveform

        ret: Tuple[Dict[str, Any], List[Tuple[str, bytes]]] = ({}, [])

        try:
            from demucs import separate

            speach_file = os.path.join(
                self._TEMP_DIR,
                "demucs",
                self.__class__._DEMUCS_MODEL,
                os.path.splitext(os.path.basename(self.file))[0],
                "vocals.wav",
            )

            max_clip_len = timedelta(minutes=15)

            if timedelta(seconds=duration_seconds) > max_clip_len:
                from pydub import AudioSegment
                audio = AudioSegment.from_file(self.file)
                muc = []
                i = 0
                while (i * max_clip_len.total_seconds()) < duration_seconds:
                    f, t = i * max_clip_len.total_seconds(), (i + 1) * max_clip_len.total_seconds()
                    s = os.path.join(self._TEMP_DIR, f"s_{i:04d}.wav")
                    audio[f * 1000:t * 1000].export(s, format="wav")
                    muc.append((
                        s,
                        os.path.join(
                            self._TEMP_DIR,
                            "demucs",
                            self.__class__._DEMUCS_MODEL,
                            os.path.splitext(os.path.basename(s))[0],
                            "vocals.wav",
                        )
                    ))
                    i += 1
            else:
                muc = [(self.file, speach_file)]

            os.makedirs(os.path.join(self._TEMP_DIR, "demucs"), exist_ok=True)
            for f, t in muc:
                separate.main([
                    "--two-stems", "vocals",
                    "-n", self.__class__._DEMUCS_MODEL,
                    "-o", os.path.join(self._TEMP_DIR, "demucs"),
                    f,
                ])
            if len(muc) > 1:
                from pydub import AudioSegment
                import shutil
                audio = None
                for f, t in muc:
                    if audio is None:
                        audio = AudioSegment.from_file(t)
                    else:
                        audio = audio + AudioSegment.from_file(t)
                    os.remove(f)
                    shutil.rmtree(os.path.dirname(t))
                os.makedirs(os.path.dirname(speach_file), exist_ok=True)
                audio.export(speach_file, format="wav")

            self._LOGGER.info("Demucs finished")
            del separate
        except Exception as e:
            self._LOGGER.exception("Demucs failed", exc_info=e)
            speach_file = self.file

        ret[0]["duration"] = duration_seconds
        ret[0]["sample_rate"] = sample_rate

        rttm_list: Optional[List] = None
        if self.__class__._DIARIZER is None:
            self.__class__._LOGGER.error("No diarizer selected. Please make sure to meet requirements")
        else:
            from uuid import uuid4
            from collections import defaultdict
            from rttm_manager import export_rttm
            segments_by_speaker = defaultdict(list)

            rttm_list = self.__class__._DIARIZER.diarize(audio_file=speach_file, temp_dir=self._TEMP_DIR)
            for rttm in rttm_list:
                label = rttm.speaker_name
                segment_start, segment_duration = rttm.turn_onset, rttm.turn_duration
                segment_end = segment_start + segment_duration
                segments_by_speaker[label].append((segment_start, segment_end))

            speaker_counts = {k: len(v) for k, v in segments_by_speaker.items()}
            speaker_durations = {k: sum(v) for k, v in
                                 ((k2, [v2e - v2s for v2s, v2e in v2]) for k2, v2 in segments_by_speaker.items())}
            self._LOGGER.info(
                f"Speaker presence: {', '.join(str(x) for x in sorted(speaker_counts.items(), key=lambda x: x[::-1], reverse=True))}"
            )
            self._LOGGER.info(
                f"Speaker duration: {', '.join(str(x) for x in sorted(speaker_durations.items(), key=lambda x: x[::-1], reverse=True))}"
            )

            _tmp_rttm_file = os.path.join(self._TEMP_DIR, f"{uuid4()}.rttm")

            export_rttm(rttms=rttm_list, file_path=_tmp_rttm_file)

            ret[0]["speaker_counts"] = speaker_counts
            ret[0]["speaker_durations"] = speaker_durations
            with open(_tmp_rttm_file, "r") as f:
                ret[1].append(("speaker.rttm", f.read().encode("utf-8")))
            os.remove(_tmp_rttm_file)

        if self.__class__._TRANSCRIPTOR is None:
            self.__class__._LOGGER.error("No transcriptor selected. Please make sure to meet requirements")
        else:
            words, lang = self.__class__._TRANSCRIPTOR.transcribe(
                audio_file=speach_file,
                model=self._TRANSCRIPTOR_MODEL,
                language=self._TRANSCRIPTOR_LANGUAGE,
            )
            self._LOGGER.info(f"Successful transcription of audio. Transcribed as language {lang}")
            ret[0]["language"] = lang
            if rttm_list is not None and len(rttm_list) > 0:
                from rttm_manager import RTTM
                from PAT.modules.speaker.util import best_word_speaker_match
                rttm_list: List[RTTM]
                self._LOGGER.info("Diarization and Transcription ready. Matching conversation to speaker data")

                speaker_idx = 0
                word_speaker: List[WordTupleSpeaker] = []
                for word in tqdm(words, unit="word", leave=False, desc="Matching words to speakers"):
                    speaker = best_word_speaker_match(word=word, speakers=rttm_list)[0]
                    word_speaker.append(
                        WordTupleSpeaker.from_word_tuple(word=word, speaker=speaker.speaker_name)
                    )
                self._LOGGER.info("Building transcription")
                # TODO: Match Person to speaker label
                transcript: List[Tuple[str, List[str]]] = []
                for ws in word_speaker:
                    if len(transcript) <= 0 or transcript[-1][0] != ws.speaker:
                        transcript.append((ws.speaker, []))
                    transcript[-1][1].append(ws.word)
                final_transcript = "\n".join(
                    f"{s}: {restore_punctuation(' '.join(t))}" for s, t in
                    tqdm(transcript, unit="sentence", leave=False, desc="Fixing punctuations in transcript")
                )
                ret[1].append(("transcript.txt", final_transcript.encode("utf-8")))

            else:
                self._LOGGER.info("Diarization not done. Creating single transcript without speaker separation")
                transcript: str = " ".join(
                    x.word.strip() for x in words if x.word is not None and len(x.word.strip()) > 0)
                self._LOGGER.info(f"Fixing punctuation in transcript")
                transcript: str = restore_punctuation(text=transcript)
                ret[1].append(("transcript.txt", transcript.encode("utf-8")))
                final_transcript = transcript

            # TODO: Maybe use LLM to fix transcription?

            if self._SUMMARIZE_MODEL is not None:
                try:
                    from PAT.utils.summerize import summarize, Templates
                    summ = summarize(
                        text=final_transcript,
                        openai_model=self._SUMMARIZE_MODEL,
                        language=lang,
                        template=Templates.Podcast
                    )
                    if summ is not None:
                        ret[1].append(("summarization.txt", summ.encode("utf-8")))
                except Exception:
                    self._LOGGER.warning("Failed to summarize transcript")
            else:
                self._LOGGER.info("No model for summarization set. Will not summarize")

        return ret

    @classmethod
    def load(cls, config: Dict[str, Any]):
        super().load(config=config)
        cls._TEMP_DIR = config.get("temp_dir", None) or cls._TEMP_DIR
        if os.path.isfile(cls._TEMP_DIR):
            raise ValueError(f"{cls._TEMP_DIR} is an existing file. Please provide a valid folder (existing or not)")
        os.makedirs(cls._TEMP_DIR, exist_ok=True)
        cls._DIARIZER = config.get("diarizer", None)
        cls._DEMUCS_MODEL = config.get("demucs_model", cls._DEMUCS_MODEL)
        if PyannoteDiarizator.is_available():
            model_path: Optional[str] = config.get("segmentation_model", None)
            optimized_data: Optional[str] = config.get("optimized_data", None)
            PyannoteDiarizator.SEGMENTATION_MODEL = model_path
            PyannoteDiarizator.OPTIMIZED_DATA = optimized_data
        cls._TRANSCRIPTOR = config.get("transcriptor", None)
        cls._TRANSCRIPTOR_MODEL = config.get("transcriptor_model", None)
        cls._SUMMARIZE_MODEL = config.get("transcription_summarize_model", None)

    @classmethod
    def config_keys(cls) -> Dict[str, Dict[str, Any]]:
        ret: Dict[str, Dict[str, Any]] = {
            "temp_dir": {
                "type": str, "required": False, "default": cls._TEMP_DIR,
                "help": "Temporary directory to store files. Eg transcoding. (Default: %(default)s)"
            },
            "demucs_model": {
                "type": str, "required": False, "default": cls._DEMUCS_MODEL,
                "help": "Model to use with demucs (https://github.com/adefossez/demucs). (Default: %(default)s)"
            },
        }
        if len(cls._AVAILABLE_DIARIZERS) > 0:
            ret["diarizer"] = {
                "type": lambda x: cls._AVAILABLE_DIARIZERS[x], "required": False,
                "default": sorted(cls._AVAILABLE_DIARIZERS)[0], "choices": cls._AVAILABLE_DIARIZERS,
                "help": f"Diarization method to use. Options: {', '.join(cls._AVAILABLE_DIARIZERS)}. (Default: %(default)s)"
            }
        if len(cls._AVAILABLE_TRANSCRIPTORS) > 0:
            ret["transcriptor"] = {
                "type": lambda x: cls._AVAILABLE_TRANSCRIPTORS[x], "required": False,
                "default": sorted(cls._AVAILABLE_TRANSCRIPTORS)[0], "choices": cls._AVAILABLE_TRANSCRIPTORS,
                "help": f"Transcription method to use. Options: {', '.join(cls._AVAILABLE_TRANSCRIPTORS)}. (Default: %(default)s)"
            }
            ret["transcriptor_model"] = {
                "type": str, "required": False, "default": "medium",
                "help": f"Transcription models. Currently see official whisper models. (Default: %(default)s)"
            }
            ret["transcriptor_language"] = {
                "type": str, "required": False, "default": None,
                "help": f"Transcription language. Omit for autodetect. (Default: %(default)s)"
            }
            ret["transcription_summarize_model"] = {
                "type": str, "required": False, "default": None,
                "help": f"OpenAI model to use to summarize transcription. "
                        f"If not provided no summarization will be performed. (Default: %(default)s)"
            }
        if PyannoteDiarizator.is_available():
            ret["segmentation_model"] = {
                "type": str, "required": False, "default": None,
                "help": f"Set path to custom trained segmentation model (Used in {PyannoteDiarizator.__name__})"
            }
            ret["optimized_data"] = {
                "type": str, "required": False, "default": None,
                "help": f"Set path to optimized data to instantiate pipeline (Used in {PyannoteDiarizator.__name__})"
            }
        return ret

    @classmethod
    def unload(cls):
        super().unload()
        if cls._PIPELINE is not None:
            try:
                cls._PIPELINE.unregister()
            except AttributeError:
                pass
            del cls._PIPELINE
