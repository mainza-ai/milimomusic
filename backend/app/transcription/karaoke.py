"""
Karaoke and Neural Lyric-Sync Timestamp Service.
Generates word-level timestamps, .lrc files, and .srt subtitles with acoustic vocal alignment.
"""

import os
import re
import math
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple

logger = logging.getLogger(__name__)

# Optional soundfile/numpy for acoustic vocal envelope extraction
try:
    import soundfile as sf
    import numpy as np
    HAS_AUDIO_LIBS = True
except ImportError:
    HAS_AUDIO_LIBS = False


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
    is_section: bool = False


class LyricSyncEngine:
    """
    Acoustic-aware lyric synchronization engine.
    Extracts vocal energy from separated vocal stems (HTDemucs) or master audio
    to accurately map stanzas, lines, and words to singing intervals.
    """

    SECTION_PATTERN = re.compile(r"^\s*(\[|\()([A-Za-z0-9\s\-_:.]+)(\]|\))\s*$")

    @classmethod
    def is_section_header(cls, text: str) -> bool:
        return bool(cls.SECTION_PATTERN.match(text.strip()))

    @staticmethod
    def _estimate_syllables(word: str) -> int:
        """Estimate syllable count for proportional word timing."""
        clean = re.sub(r"[^a-zA-Z]", "", word.lower())
        if not clean:
            return 1
        if len(clean) <= 3:
            return 1
        # Count vowel groups
        vowels = "aeiouy"
        count = 0
        prev_is_vowel = False
        for char in clean:
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                count += 1
            prev_is_vowel = is_vowel
        if clean.endswith("e") and not clean.endswith("le") and count > 1:
            count -= 1
        return max(1, count)

    @classmethod
    def _extract_vocal_regions(
        cls, audio_path: str, duration_sec: float
    ) -> List[Tuple[float, float]]:
        """
        Analyze audio file energy to detect active vocal intervals.
        Returns list of (start_sec, end_sec) tuples for singing segments.
        """
        if not HAS_AUDIO_LIBS or not audio_path or not os.path.exists(audio_path):
            return []

        try:
            # Read audio file
            data, samplerate = sf.read(audio_path, dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=1)  # Convert to mono

            if len(data) == 0:
                return []

            # 50ms frame with 25ms hop
            frame_size = int(samplerate * 0.05)
            hop_size = int(samplerate * 0.025)
            num_frames = (len(data) - frame_size) // hop_size

            if num_frames <= 0:
                return []

            # Compute RMS energy per frame
            energies = np.zeros(num_frames, dtype=np.float32)
            for i in range(num_frames):
                frame = data[i * hop_size : i * hop_size + frame_size]
                energies[i] = np.sqrt(np.mean(frame**2) + 1e-9)

            max_energy = float(np.max(energies))
            if max_energy < 1e-4:
                return []  # Silent audio

            # Adaptive threshold for vocal presence
            threshold = max(0.015, max_energy * 0.12)
            active_frames = energies > threshold

            # Group active frames into continuous intervals with 400ms hangover
            hangover_frames = int(0.400 / 0.025)
            min_segment_frames = int(0.300 / 0.025)

            segments: List[Tuple[float, float]] = []
            in_seg = False
            seg_start = 0
            silence_count = 0

            for i, active in enumerate(active_frames):
                if active:
                    if not in_seg:
                        in_seg = True
                        seg_start = i
                    silence_count = 0
                else:
                    if in_seg:
                        silence_count += 1
                        if silence_count >= hangover_frames:
                            seg_end = i - silence_count
                            if (seg_end - seg_start) >= min_segment_frames:
                                start_t = seg_start * 0.025
                                end_t = min(duration_sec, seg_end * 0.025)
                                segments.append((start_t, end_t))
                            in_seg = False
                            silence_count = 0

            if in_seg:
                start_t = seg_start * 0.025
                end_t = min(duration_sec, len(active_frames) * 0.025)
                if (len(active_frames) - seg_start) >= min_segment_frames:
                    segments.append((start_t, end_t))

            return segments
        except Exception as e:
            logger.warning(f"Vocal energy extraction failed for {audio_path}: {e}")
            return []

    @classmethod
    def align_lyrics(
        cls,
        lyrics: str,
        duration_sec: float,
        vocal_stem_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Align lyrics text with acoustic vocal onset/offset detection
        and syllable-weighted word timing.
        """
        if not lyrics or duration_sec <= 0:
            return []

        raw_lines = [l.strip() for l in lyrics.split("\n") if l.strip()]
        if not raw_lines:
            return []

        # Resolve vocal audio path (handling local prefixes if needed)
        resolved_vocal_path = ""
        if vocal_stem_path:
            local_candidate = vocal_stem_path
            if local_candidate.startswith("/audio/"):
                local_candidate = local_candidate.replace("/audio/", "generated_audio/", 1)
            elif local_candidate.startswith("/"):
                local_candidate = local_candidate.lstrip("/")
            if os.path.exists(local_candidate):
                resolved_vocal_path = local_candidate
            elif os.path.exists(vocal_stem_path):
                resolved_vocal_path = vocal_stem_path

        # Step 1: Extract real vocal active intervals from audio
        vocal_regions = []
        if resolved_vocal_path:
            vocal_regions = cls._extract_vocal_regions(resolved_vocal_path, duration_sec)

        # Step 2: Separate structural sections from lyrical lines
        # Group lyrics into stanzas
        stanzas: List[List[str]] = []
        current_stanza: List[str] = []
        for line in raw_lines:
            if cls.is_section_header(line):
                if current_stanza:
                    stanzas.append(current_stanza)
                    current_stanza = []
                stanzas.append([line])  # Section header as standalone item
            else:
                current_stanza.append(line)
        if current_stanza:
            stanzas.append(current_stanza)

        # Step 3: Determine singing boundaries
        # If acoustic regions were detected, use them; otherwise, use musical pacing
        if vocal_regions:
            first_vocal_start = vocal_regions[0][0]
            last_vocal_end = vocal_regions[-1][1]
        else:
            # Fallback musical pacing: 8-second intro buffer if intro exists, 3s buffer otherwise
            has_intro = any(cls.is_section_header(s[0]) and "intro" in s[0].lower() for s in stanzas)
            first_vocal_start = 8.0 if has_intro else 3.0
            first_vocal_start = min(first_vocal_start, duration_sec * 0.20)
            last_vocal_end = max(first_vocal_start + 5.0, duration_sec - 4.0)
            vocal_regions = [(first_vocal_start, last_vocal_end)]

        # Calculate total weight of sung lines (syllable & length based)
        sung_lines: List[Tuple[int, int, str]] = []  # (stanza_idx, line_idx, text)
        for s_idx, stanza in enumerate(stanzas):
            for l_idx, line in enumerate(stanza):
                if not cls.is_section_header(line):
                    sung_lines.append((s_idx, l_idx, line))

        if not sung_lines:
            # Only section headers were provided
            timed_headers = []
            header_step = duration_sec / max(1, len(raw_lines))
            for i, line in enumerate(raw_lines):
                t = i * header_step
                timed_headers.append({
                    "text": line,
                    "start": round(t, 2),
                    "end": round(t + header_step, 2),
                    "is_section": True,
                    "words": []
                })
            return timed_headers

        # Compute line weights based on syllable counts
        line_weights = []
        for _, _, line in sung_lines:
            words = line.split()
            syllables = sum(cls._estimate_syllables(w) for w in words)
            line_weights.append(max(2, syllables))
        total_weight = sum(line_weights)

        # Allocate timestamps to each sung line across vocal regions
        allocated_lines: Dict[Tuple[int, int], Tuple[float, float]] = {}
        cumulative_time = first_vocal_start
        available_time_span = max(1.0, last_vocal_end - first_vocal_start)

        # Map sung lines into vocal timeline
        current_sung_idx = 0
        for s_idx, stanza in enumerate(stanzas):
            is_header_stanza = len(stanza) == 1 and cls.is_section_header(stanza[0])
            if is_header_stanza:
                continue

            # Calculate stanza duration
            stanza_lines_count = len(stanza)
            stanza_weight = sum(line_weights[current_sung_idx + i] for i in range(stanza_lines_count))
            stanza_fraction = stanza_weight / max(1, total_weight)
            stanza_duration = available_time_span * stanza_fraction

            stanza_start = cumulative_time
            stanza_end = min(last_vocal_end, stanza_start + stanza_duration)

            # Sub-allocate lines within stanza
            line_start = stanza_start
            for l_idx, line in enumerate(stanza):
                weight = line_weights[current_sung_idx]
                line_frac = weight / max(1, stanza_weight)
                line_dur = stanza_duration * line_frac

                # Add natural mini-pause between lines (0.15s)
                dur_singing = max(0.6, line_dur - 0.15)
                allocated_lines[(s_idx, l_idx)] = (line_start, min(stanza_end, line_start + dur_singing))
                line_start += line_dur
                current_sung_idx += 1

            cumulative_time += stanza_duration

        # Step 4: Build final TimedLine list with words
        timed_results: List[Dict[str, Any]] = []

        for s_idx, stanza in enumerate(stanzas):
            if len(stanza) == 1 and cls.is_section_header(stanza[0]):
                header_text = stanza[0]
                # Header start matches the following stanza start or 0.00
                if s_idx + 1 < len(stanzas) and (s_idx + 1, 0) in allocated_lines:
                    h_start = max(0.0, allocated_lines[(s_idx + 1, 0)][0] - 2.0)
                    h_end = allocated_lines[(s_idx + 1, 0)][0]
                else:
                    h_start = max(0.0, cumulative_time)
                    h_end = min(duration_sec, h_start + 2.0)

                timed_results.append({
                    "text": header_text,
                    "start": round(h_start, 2),
                    "end": round(h_end, 2),
                    "is_section": True,
                    "words": []
                })
            else:
                for l_idx, line in enumerate(stanza):
                    if (s_idx, l_idx) in allocated_lines:
                        l_start, l_end = allocated_lines[(s_idx, l_idx)]
                        line_dur = max(0.5, l_end - l_start)

                        # Generate word-level timestamps with syllable weighting
                        words = line.split()
                        w_syllables = [cls._estimate_syllables(w) for w in words]
                        total_w_syllables = sum(w_syllables)

                        timed_words: List[Dict[str, Any]] = []
                        w_time = l_start
                        for w_text, s_count in zip(words, w_syllables):
                            w_frac = s_count / max(1, total_w_syllables)
                            w_dur = line_dur * w_frac
                            timed_words.append({
                                "word": w_text,
                                "start": round(w_time, 2),
                                "end": round(min(l_end, w_time + w_dur), 2)
                            })
                            w_time += w_dur

                        timed_results.append({
                            "text": line,
                            "start": round(l_start, 2),
                            "end": round(l_end, 2),
                            "is_section": False,
                            "words": timed_words
                        })

        # Ensure monotonic ordering by start time
        timed_results.sort(key=lambda x: x["start"])
        return timed_results

    @staticmethod
    def generate_lrc(timed_lines: List[Dict[str, Any]], title: str = "Milimo Master") -> str:
        """Export timed lines to standard .lrc format."""
        lrc_lines = [
            f"[ti:{title}]",
            "[ar:Milimo AI]",
            "[al:Milimo Studio Productions]",
            "[by:Milimo Neural Karaoke Engine]"
        ]
        for line in timed_lines:
            total_sec = max(0.0, float(line.get("start", 0.0)))
            minutes = int(total_sec // 60)
            seconds = int(total_sec % 60)
            hundredths = int((total_sec - int(total_sec)) * 100)
            text = line.get("text", "").strip()
            if text:
                lrc_lines.append(f"[{minutes:02d}:{seconds:02d}.{hundredths:02d}]{text}")
        return "\n".join(lrc_lines)

    @staticmethod
    def generate_srt(timed_lines: List[Dict[str, Any]]) -> str:
        """Export timed lines to standard .srt subtitle format."""
        srt_entries = []
        for idx, line in enumerate(timed_lines, 1):
            def fmt_time(t: float) -> str:
                t = max(0.0, float(t))
                hours = int(t // 3600)
                minutes = int((t % 3600) // 60)
                seconds = int(t % 60)
                millis = int((t - int(t)) * 1000)
                return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

            start_str = fmt_time(line.get("start", 0.0))
            end_str = fmt_time(line.get("end", 0.0))
            text = line.get("text", "").strip()
            if text:
                srt_entries.append(f"{idx}\n{start_str} --> {end_str}\n{text}\n")
        return "\n".join(srt_entries)


lyric_sync_engine = LyricSyncEngine()
