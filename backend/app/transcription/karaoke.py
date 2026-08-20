"""
Karaoke and Lyric-Sync Timestamp Service.
Generates word-level timestamps, .lrc files, and .srt subtitles for live karaoke playback.
"""

import os
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Any


@dataclass
class TimedWord:
    word: str
    start: float
    end: float


@dataclass
class TimedLine:
    text: str
    start: float
    end: float
    words: List[TimedWord]


class LyricSyncEngine:
    @staticmethod
    def align_lyrics(lyrics: str, duration_sec: float) -> List[Dict[str, Any]]:
        """
        Align lyrics text with estimated or forced timestamps across song duration.
        """
        if not lyrics:
            return []

        raw_lines = [l.strip() for l in lyrics.split("\n") if l.strip()]
        if not raw_lines:
            return []

        # Remove header brackets from spoken calculation or mark them
        timed_lines = []
        time_per_line = duration_sec / max(1, len(raw_lines))
        current_time = 0.0

        for line in raw_lines:
            line_duration = time_per_line
            words = line.split()
            time_per_word = line_duration / max(1, len(words))

            timed_words = []
            word_time = current_time
            for w in words:
                timed_words.append({
                    "word": w,
                    "start": round(word_time, 2),
                    "end": round(word_time + time_per_word, 2)
                })
                word_time += time_per_word

            timed_lines.append({
                "text": line,
                "start": round(current_time, 2),
                "end": round(current_time + line_duration, 2),
                "words": timed_words
            })
            current_time += line_duration

        return timed_lines

    @staticmethod
    def generate_lrc(timed_lines: List[Dict[str, Any]]) -> str:
        """Export timed lines to standard .lrc format."""
        lrc_lines = ["[ti:Milimo Music Master]", "[ar:Milimo AI]", "[al:Milimo v2 DAW]"]
        for line in timed_lines:
            total_sec = line["start"]
            minutes = int(total_sec // 60)
            seconds = int(total_sec % 60)
            hundredths = int((total_sec - int(total_sec)) * 100)
            lrc_lines.append(f"[{minutes:02d}:{seconds:02d}.{hundredths:02d}]{line['text']}")
        return "\n".join(lrc_lines)

    @staticmethod
    def generate_srt(timed_lines: List[Dict[str, Any]]) -> str:
        """Export timed lines to standard .srt subtitle format."""
        srt_entries = []
        for idx, line in enumerate(timed_lines, 1):
            def fmt_time(t: float) -> str:
                hours = int(t // 3600)
                minutes = int((t % 3600) // 60)
                seconds = int(t % 60)
                millis = int((t - int(t)) * 1000)
                return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

            start_str = fmt_time(line["start"])
            end_str = fmt_time(line["end"])
            srt_entries.append(f"{idx}\n{start_str} --> {end_str}\n{line['text']}\n")
        return "\n".join(srt_entries)


lyric_sync_engine = LyricSyncEngine()
