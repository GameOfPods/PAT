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

def restore_punctuation(text: str, model: str = "kredor/punctuate-all"):
    import logging
    try:
        from deepmultilingualpunctuation import PunctuationModel

        model = PunctuationModel(model=model)
        prediction = model.restore_punctuation(text=text)
        return prediction
    except ImportError as e:
        logging.getLogger(__name__).exception("Punctuation requirements not met. Check requirements", exc_info=e)
        return text
