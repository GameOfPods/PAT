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
from PAT.modules.speaker.speakermatching import SpeakerMatching
from PAT.modules.speaker.transcriptors import Transcriptor, WordTupleSpeaker
from PAT.utils.punctuation import restore_punctuation
from PAT.utils.summerize import LLM as LLMService


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
    _SPEAKER_MATCHER: Optional[SpeakerMatching] = None
    _SUMMARIZE_SERVICE: Optional[LLMService] = None

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
                self._LOGGER.info(
                    f"Splitting audio in {len(muc)} {max_clip_len} long snippets for demucs to save on ram usage"
                )
            else:
                muc = [(self.file, speach_file)]
                self._LOGGER.info(f"Audio clip is shorter than {max_clip_len}. Will not split audio for demucs")

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
            from rttm_manager import export_rttm, RTTM
            segments_by_speaker = defaultdict(list)

            rttm_list: List[RTTM] = self.__class__._DIARIZER.diarize(audio_file=speach_file, temp_dir=self._TEMP_DIR)
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

            all_speakers: Dict[str, Optional[Tuple[str, float]]] = {x.speaker_name: None for x in rttm_list}

            def speaker_str(_s):
                return f'{all_speakers[_s][0]}' if _s in all_speakers else _s

            if self._SPEAKER_MATCHER is not None:
                from pydub import AudioSegment
                from io import BytesIO
                audio = AudioSegment.from_file(speach_file)
                for s in all_speakers.keys():
                    speaker_times: List[Tuple[Optional[float], Optional[float]]] = []

                    for rttm in (x for x in rttm_list if x.speaker_name == s):
                        st, en = rttm.turn_onset, rttm.turn_onset + rttm.turn_duration

                        def overlap(_x: RTTM):
                            _x1, _x2 = _x.turn_onset, _x.turn_onset + _x.turn_duration
                            return st <= _x1 <= en or st <= _x2 <= en or _x1 <= st <= _x2 or _x1 <= en <= _x2

                        if not any(overlap(_x=x) for x in rttm_list if x.speaker_name != s):
                            speaker_times.append((st, en))
                        speaker_times = [x for x in speaker_times if not any(y is None for y in x)]
                        if sum(abs(e - s) for s, e in speaker_times) > 10:
                            break

                    if len(speaker_times) > 0:
                        speaker_audio = audio[speaker_times[0][0] * 1000:speaker_times[0][1] * 1000]
                        for speaker_time in speaker_times[1:]:
                            speaker_audio = speaker_audio + audio[speaker_time[0] * 1000:speaker_time[1] * 1000]
                        with BytesIO() as output:
                            speaker_audio.export(output, format="mp3")
                            output.seek(0)
                            pred_speaker = self._SPEAKER_MATCHER.match_speaker(audio=output)
                            if pred_speaker:
                                all_speakers[s] = pred_speaker

                self._LOGGER.info(", ".join(f"{k} might be {v1} ({v2:.4f})" for k, (v1, v2) in all_speakers.items()))
                speaker_confidence_t = float(os.environ.get("SPEAKER_CONFIDENCE_VALUE", 0.5))
                if speaker_confidence_t > 0:
                    self._LOGGER.info(
                        f"Removing all speaker mappings that have confidence lower than {speaker_confidence_t}"
                    )
                    all_speakers = {x: y if y[1] > speaker_confidence_t else None for x, y in all_speakers.items()}

                if len(set(all_speakers.values())) != len(all_speakers):
                    from collections import Counter
                    self._LOGGER.error("Two or more speakers matched to the same sample. This should not happen")
                    c = Counter(all_speakers.values())
                    for k, v in all_speakers.items():
                        if c[v] >= 2:
                            all_speakers[k] = f"{v[0]}-{c[v]}", v[1]
                            c[v] -= 1

            all_speakers_keys = all_speakers.keys()
            all_speakers = {x: y for x, y in all_speakers.items() if y is not None}

            self._LOGGER.info(f"Final mapping: {', '.join(f'{s} -> {speaker_str(s)}' for s in all_speakers_keys)}")

            ret[0]["speaker_mapping"] = {s: speaker_str(s) for s in all_speakers_keys}

            if "speaker_counts" in ret[0]:
                ret[0]["speaker_counts_o"] = ret[0]["speaker_counts"]
                ret[0]["speaker_counts"] = {speaker_str(k): v for k, v in ret[0]["speaker_counts_o"].items()}
            if "speaker_durations" in ret[0]:
                ret[0]["speaker_durations_o"] = ret[0]["speaker_durations"]
                ret[0]["speaker_durations"] = {speaker_str(k): v for k, v in ret[0]["speaker_durations_o"].items()}

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

            self._LOGGER.info("Creating naive transcript without using diarization")
            transcript: str = " ".join(x.word.strip() for x in words if x.word is not None and len(x.word.strip()) > 0)
            self._LOGGER.info(f"Fixing punctuation in transcript")
            transcript: str = restore_punctuation(text=transcript)
            ret[1].append(("transcript.txt", transcript.encode("utf-8")))

            if rttm_list is not None and len(rttm_list) > 0:
                from rttm_manager import RTTM
                from PAT.modules.speaker.util import best_word_speaker_match
                rttm_list: List[RTTM]
                self._LOGGER.info("Diarization and Transcription ready. Matching conversation to speaker data")

                word_speaker: List[WordTupleSpeaker] = []
                for word in tqdm(words, unit="word", leave=False, desc="Matching words to speakers"):
                    speaker = best_word_speaker_match(word=word, speakers=rttm_list)[0]
                    word_speaker.append(
                        WordTupleSpeaker.from_word_tuple(word=word, speaker=speaker.speaker_name)
                    )
                self._LOGGER.info("Building transcription")
                transcript: List[Tuple[WordTupleSpeaker, List[str]]] = []
                for ws in word_speaker:
                    if len(transcript) <= 0 or transcript[-1][0].speaker != ws.speaker or abs(
                            transcript[-1][0].end - ws.start) > 10:
                        transcript.append((ws, []))
                    transcript[-1][-1].append(ws.word)

                final_transcript = "\n".join(
                    f"{timedelta(seconds=s.start)} {speaker_str(s.speaker)}: {restore_punctuation(' '.join(t))}"
                    for s, t in tqdm(transcript, unit="section", leave=False, desc="Fixing punctuations in transcript")
                )
                ret[1].append(("transcript_speaker.txt", final_transcript.encode("utf-8")))

            else:
                self._LOGGER.info("Diarization not done. Using naive transcript as final transcript")
                final_transcript = transcript

            # TODO: Maybe use LLM to fix transcription?

            if self._SUMMARIZE_SERVICE is not None and self._SUMMARIZE_MODEL is not None:
                try:
                    from PAT.utils.summerize import summarize, Templates
                    self._LOGGER.info("Creating summary of final transcript")
                    summ = summarize(
                        text=final_transcript,
                        model=self._SUMMARIZE_MODEL,
                        lang=lang,
                        template=Templates.Podcast,
                        llm=self._SUMMARIZE_SERVICE,
                    )
                    if summ is not None:
                        ret[1].append(("summarization.txt", summ.encode("utf-8")))
                except Exception as e:
                    self._LOGGER.error(f"Failed to summarize transcript: {e}")
            else:
                self._LOGGER.info("No service/model for summarization set. Will not summarize")

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
        cls._SUMMARIZE_SERVICE = config.get("transcription_summarize_service", LLMService.OpenAI)
        if "transcription_summarize_service" in config:
            try:
                cls._SUMMARIZE_SERVICE = [
                    x for x in LLMService if x.name == config["transcription_summarize_service"]
                ][0]
            except IndexError:
                cls._LOGGER.error(
                    f"Could not find Summarization LLM service with name {config['transcription_summarize_service']}"
                )
        if "speaker_samples" in config and config["speaker_samples"] is not None and len(config["speaker_samples"]) > 0:
            cls._SPEAKER_MATCHER = SpeakerMatching(
                {os.path.splitext(os.path.basename(x))[0]: x for x in config["speaker_samples"]}
            )

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
                "help": f"Diarization method to use. Options: {', '.join(cls._AVAILABLE_DIARIZERS)}. "
                        f"(Default: %(default)s)"
            }
            if SpeakerMatching.is_available():
                def _valid_file(x):
                    import os
                    if os.path.exists(x) and os.path.isfile(x):
                        return x

                ret["speaker_samples"] = {
                    "type": _valid_file, "required": False, "default": None, "nargs": "+",
                    "help": "Provide sample audio files to match the diarized speakers against. "
                            "Should be formatted as /path/to/file/<name_of_speaker>.ext"
                }
        if len(cls._AVAILABLE_TRANSCRIPTORS) > 0:
            ret["transcriptor"] = {
                "type": lambda x: cls._AVAILABLE_TRANSCRIPTORS[x], "required": False,
                "default": sorted(cls._AVAILABLE_TRANSCRIPTORS)[0], "choices": cls._AVAILABLE_TRANSCRIPTORS,
                "help": f"Transcription method to use. Choices: %(choices)s. (Default: %(default)s)"
            }
            ret["transcriptor_model"] = {
                "type": str, "required": False, "default": "medium",
                "help": f"Transcription models. Currently see official whisper models. (Default: %(default)s)"
            }
            ret["transcriptor_language"] = {
                "type": str, "required": False, "default": None,
                "help": f"Transcription language. Omit for autodetect. (Default: %(default)s)"
            }
            ret["transcription_summarize_service"] = {
                "type": str, "choices": [x.name for x in LLMService], "required": False,
                "default": LLMService.OpenAI.name,
                "help": f"Service to use for summarization tasks. Choices: %(choices)s. (Default: %(default)s)"
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
