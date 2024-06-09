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
from typing import Tuple

from PAT.modules.speaker.diarizators import Diarizators


class NemoDiarizator(Diarizators):
    _LOGGER = logging.getLogger("PyannoteDiarizator")

    @classmethod
    def is_available(cls) -> bool:
        try:
            from pydub import AudioSegment
            from nemo.collections.asr.models.msdd_models import NeuralDiarizer
            import torch
        except ImportError:
            return False
        return True

    @classmethod
    def diarize(cls, audio_file: str, temp_dir: str, device: str = None):
        from pydub import AudioSegment
        from nemo.collections.asr.models.msdd_models import NeuralDiarizer
        from rttm_manager import import_rttm
        sound = AudioSegment.from_file(audio_file).set_channels(1)
        audio_file_mono = f"{audio_file}.mono.wav"
        sound.export(audio_file_mono, format="wav")

        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model_cfg, out_dir = cls._create_config(audio_file=audio_file_mono, temp_dir=temp_dir)
        model = NeuralDiarizer(cfg=model_cfg).to(device)
        model.diarize()

        del model
        os.remove(model_cfg.diarizer.manifest_filepath)
        if device == "cuda":
            import torch
            torch.cuda.empty_cache()

        rttm_f = os.path.join(out_dir, "pred_rttms",
                              f"{'.'.join(os.path.basename(audio_file_mono).split('.')[:-1])}.rttm")
        with open(rttm_f, "r") as f:
            import re
            rttm_data = f.read()
            rttm_data = re.sub(r" +", " ", rttm_data)
            rttm_data = "\n".join(x.strip() for x in rttm_data.splitlines() if len(x.strip()) > 0)
        with open(rttm_f, "w") as f:
            f.write(rttm_data)

        ret = import_rttm(rttm_f)

        import shutil
        shutil.rmtree(model_cfg.diarizer.out_dir)

        return ret

    @classmethod
    def _create_config(cls, audio_file: str, temp_dir: str) -> Tuple["OmegaConf", str]:
        import os
        import io
        from uuid import uuid4
        import json
        from omegaconf import OmegaConf
        import requests

        out_dir = os.path.join(temp_dir, "nemo_data")

        domain = os.environ.get("NEMO_DOMAIN", "telephonic")
        config_io = io.StringIO(
            requests.get(
                f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_{domain}.yaml"
            ).text
        )

        config = OmegaConf.load(config_io)

        meta = {
            "audio_filepath": audio_file,
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "rttm_filepath": None,
            "uem_filepath": None,
        }

        _manifest_file = os.path.join(temp_dir, f"input_manifest_{uuid4()}.json")
        with open(_manifest_file, "w") as fp:
            json.dump(meta, fp)
            fp.write("\n")

        pretrained_vad = "vad_multilingual_marblenet"
        pretrained_speaker_model = "titanet_large"
        config.num_workers = 0
        config.diarizer.manifest_filepath = os.path.join(temp_dir, _manifest_file)
        config.diarizer.out_dir = out_dir
        config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
        config.diarizer.oracle_vad = False
        config.diarizer.clustering.parameters.oracle_num_speakers = False
        config.diarizer.vad.model_path = pretrained_vad
        config.diarizer.vad.parameters.onset = 0.8
        config.diarizer.vad.parameters.offset = 0.6
        config.diarizer.vad.parameters.pad_offset = -0.05
        config.diarizer.msdd_model.model_path = f"diar_msdd_{domain}"

        return config, out_dir
