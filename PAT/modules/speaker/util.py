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


from typing import List


def word_speaker_match(word, speaker) -> float:
    speaker_start, speaker_end = speaker.turn_onset, speaker.turn_onset + speaker.turn_duration
    if word.start > speaker_end or word.end < speaker_start:
        return 0
    common_start = max(word.start, speaker.turn_onset)
    common_end = min(word.end, speaker.turn_onset + speaker.turn_duration)
    if common_start == common_end:
        return 0
    return (common_end - common_start) / (word.end - word.start)


def best_word_speaker_match(word, speakers: List) -> List:
    # TODO: Optimize
    return sorted(speakers, key=lambda x: (word_speaker_match(word, x), x.turn_onset), reverse=True)
