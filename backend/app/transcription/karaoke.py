"""
Karaoke and Neural Lyric-Sync Timestamp Service.
Generates word-level timestamps, .lrc files, and .srt subtitles with acoustic vocal alignment.
Implements a 3-tier architecture:
  Tier 1: Neural Forced Alignment (TorchAudio MMS_FA) on isolated vocal stem.
  Tier 2: Multi-interval Adaptive Acoustic VAD (preserving instrumental gaps).
  Tier 3: Syllable-weighted proportional fallback (offline / synthetic).
"""

import os
import re
import math
import logging
import threading
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

# Optional PyTorch & TorchAudio for Neural Forced Alignment (MMS_FA)
try:
    import torch
    import torchaudio
    HAS_TORCH_AUDIO = True
except ImportError:
    HAS_TORCH_AUDIO = False


_mms_fa_lock = threading.Lock()
_mms_fa_bundle = None
_mms_fa_model = None
_mms_fa_tokenizer = None
_mms_fa_aligner = None


def get_mms_fa_components():
    """Lazily and thread-safely load the TorchAudio MMS_FA model and aligner."""
    global _mms_fa_bundle, _mms_fa_model, _mms_fa_tokenizer, _mms_fa_aligner
    if _mms_fa_model is not None:
        return _mms_fa_bundle, _mms_fa_model, _mms_fa_tokenizer, _mms_fa_aligner

    with _mms_fa_lock:
        if _mms_fa_model is None:
            if not HAS_TORCH_AUDIO:
                raise RuntimeError("torchaudio is not available for MMS_FA")
            bundle = torchaudio.pipelines.MMS_FA
            model = bundle.get_model()
            model.eval()
            tokenizer = bundle.get_tokenizer()
            aligner = bundle.get_aligner()
            _mms_fa_bundle = bundle
            _mms_fa_model = model
            _mms_fa_tokenizer = tokenizer
            _mms_fa_aligner = aligner
            logger.info("TorchAudio MMS_FA Neural Forced Aligner loaded successfully.")
    return _mms_fa_bundle, _mms_fa_model, _mms_fa_tokenizer, _mms_fa_aligner


def _resolve_audio_file(path: Optional[str]) -> Optional[str]:
    """Resolve any audio file candidate path to an existing local file."""
    if not path or not isinstance(path, str):
        return None
    candidates = [
        path,
        os.path.abspath(path),
        os.path.join("backend", path.lstrip("/")),
        path.replace("/audio/", "backend/generated_audio/"),
        path.replace("/audio/", "generated_audio/"),
        os.path.join("backend/generated_audio", os.path.basename(path)),
        os.path.join("backend/generated_audio/stems", os.path.basename(path)),
        os.path.join("generated_audio", os.path.basename(path)),
        os.path.join("generated_audio/stems", os.path.basename(path)),
    ]
    for cand in candidates:
        if os.path.exists(cand) and os.path.isfile(cand):
            return cand
    return None


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
    Extracts vocal energy from separated vocal stems (BS-Roformer/HTDemucs) or master audio
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
    def _align_neural_mms_fa(
        cls, lyrics: str, duration_sec: float, vocal_path: str
    ) -> List[Dict[str, Any]]:
        """
        Tier 1: True acoustic forced alignment with TorchAudio MMS_FA.
        Maps each word and line directly to acoustic emission on the vocal stem.
        """
        bundle, model, tokenizer, aligner = get_mms_fa_components()

        # Load audio into PyTorch tensor
        waveform, sr = torchaudio.load(vocal_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)

        # Parse lyrics lines & words while preserving section header structure
        raw_lines = [l.strip() for l in lyrics.split("\n") if l.strip()]
        if not raw_lines:
            return []

        parsed_structure: List[Dict[str, Any]] = []
        all_words_clean: List[str] = []
        word_tracking: List[Tuple[int, int, str, str]] = []  # (line_idx, word_idx, orig_word, clean_word)

        for l_idx, line in enumerate(raw_lines):
            if cls.is_section_header(line):
                parsed_structure.append({
                    "line_idx": l_idx,
                    "text": line,
                    "is_section": True,
                    "words": []
                })
            else:
                words = line.split()
                parsed_structure.append({
                    "line_idx": l_idx,
                    "text": line,
                    "is_section": False,
                    "words": words
                })
                for w_idx, w in enumerate(words):
                    clean = re.sub(r"[^a-z']", "", w.lower())
                    if clean:
                        all_words_clean.append(clean)
                        word_tracking.append((l_idx, w_idx, w, clean))

        if not all_words_clean:
            return []

        # Run acoustic emission
        with torch.inference_mode():
            emission, _ = model(waveform)

        tokens = tokenizer(all_words_clean)
        # aligner aligns token sequences to emission frames
        token_spans = aligner(emission[0], tokens)

        num_frames = emission.size(1)
        ratio = waveform.size(1) / max(1, num_frames)

        timed_words_by_line: Dict[int, List[Dict[str, Any]]] = {}
        for (l_idx, w_idx, orig_w, _), spans in zip(word_tracking, token_spans):
            if not spans:
                continue
            w_start = round(float(ratio * spans[0].start / bundle.sample_rate), 2)
            w_end = round(float(ratio * spans[-1].end / bundle.sample_rate), 2)
            w_end = max(w_end, round(w_start + 0.08, 2))
            timed_words_by_line.setdefault(l_idx, []).append({
                "word": orig_w,
                "start": w_start,
                "end": w_end
            })

        # Assemble lines
        final_timed: List[Dict[str, Any]] = []
        for item in parsed_structure:
            l_idx = item["line_idx"]
            if item["is_section"]:
                final_timed.append({
                    "text": item["text"],
                    "is_section": True,
                    "start": 0.0,
                    "end": 0.0,
                    "words": []
                })
            else:
                t_words = timed_words_by_line.get(l_idx, [])
                if t_words:
                    l_start = t_words[0]["start"]
                    l_end = t_words[-1]["end"]
                    final_timed.append({
                        "text": item["text"],
                        "is_section": False,
                        "start": l_start,
                        "end": l_end,
                        "words": t_words
                    })

        if not final_timed:
            return []

        # Deconflict section headers: cue them strictly during the gap preceding the section
        for i, line in enumerate(final_timed):
            if line["is_section"]:
                next_sung = next((l for l in final_timed[i + 1:] if not l["is_section"]), None)
                prev_sung = next((l for l in reversed(final_timed[:i]) if not l["is_section"]), None)
                if next_sung:
                    if prev_sung:
                        line["start"] = round(prev_sung["end"], 2)
                        line["end"] = round(next_sung["start"], 2)
                    else:
                        line["start"] = 0.0
                        line["end"] = round(next_sung["start"], 2)
                else:
                    line["start"] = prev_sung["end"] if prev_sung else 0.0
                    line["end"] = round(min(duration_sec, line["start"] + 1.5), 2)

        # Sort and ensure non-overlapping, strictly monotonic ordering
        final_timed.sort(key=lambda x: x["start"])
        return final_timed

    @classmethod
    def _extract_vocal_regions(
        cls, audio_path: str, duration_sec: float
    ) -> List[Tuple[float, float]]:
        """
        Analyze audio file energy to detect active vocal intervals.
        Returns list of (start_sec, end_sec) tuples for singing segments.
        Uses adaptive percentile thresholding so quiet or dynamic stems are detected.
        """
        if not HAS_AUDIO_LIBS or not audio_path or not os.path.exists(audio_path):
            return []

        try:
            data, samplerate = sf.read(audio_path, dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=1)  # Mono

            if len(data) == 0:
                return []

            # 50ms frame with 25ms hop
            frame_size = int(samplerate * 0.05)
            hop_size = int(samplerate * 0.025)
            num_frames = (len(data) - frame_size) // hop_size

            if num_frames <= 0:
                return []

            energies = np.zeros(num_frames, dtype=np.float32)
            for i in range(num_frames):
                frame = data[i * hop_size : i * hop_size + frame_size]
                energies[i] = np.sqrt(np.mean(frame**2) + 1e-9)

            max_energy = float(np.max(energies))
            if max_energy < 1e-5:
                return []

            # Dynamic adaptive threshold based on 75th percentile
            p75 = float(np.percentile(energies, 75))
            threshold = max(1e-4, p75 * 0.25)
            active_frames = energies > threshold

            # Group active frames with 400ms hangover and 300ms min segment
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
    def _align_acoustic_vad(
        cls,
        lyrics: str,
        duration_sec: float,
        vocal_regions: List[Tuple[float, float]]
    ) -> List[Dict[str, Any]]:
        """
        Tier 2: Multi-interval acoustic VAD allocation.
        Distributes stanzas across distinct singing intervals, preserving instrumental gaps.
        """
        raw_lines = [l.strip() for l in lyrics.split("\n") if l.strip()]
        if not raw_lines:
            return []

        # Group lyrics into stanzas
        stanzas: List[List[str]] = []
        current_stanza: List[str] = []
        for line in raw_lines:
            if cls.is_section_header(line):
                if current_stanza:
                    stanzas.append(current_stanza)
                    current_stanza = []
                stanzas.append([line])
            else:
                current_stanza.append(line)
        if current_stanza:
            stanzas.append(current_stanza)

        lyrical_stanzas = [s for s in stanzas if not (len(s) == 1 and cls.is_section_header(s[0]))]
        if not lyrical_stanzas:
            # Only section headers
            return cls._align_heuristic(lyrics, duration_sec)

        # Map lyrical stanzas to vocal regions
        # If we have regions, distribute stanzas into regions proportionately
        allocated_stanzas: Dict[int, Tuple[float, float]] = {}
        if vocal_regions:
            num_regions = len(vocal_regions)
            num_stanzas = len(lyrical_stanzas)
            for s_i, stanza in enumerate(lyrical_stanzas):
                # Map stanza index to vocal region
                r_idx = min(num_regions - 1, int(s_i * num_regions / num_stanzas))
                r_start, r_end = vocal_regions[r_idx]
                allocated_stanzas[s_i] = (r_start, r_end)
        else:
            return cls._align_heuristic(lyrics, duration_sec)

        # Allocate lines within each stanza
        timed_results: List[Dict[str, Any]] = []
        curr_lyric_s_idx = 0

        for s_idx, stanza in enumerate(stanzas):
            if len(stanza) == 1 and cls.is_section_header(stanza[0]):
                timed_results.append({
                    "text": stanza[0],
                    "is_section": True,
                    "start": 0.0,
                    "end": 0.0,
                    "words": []
                })
            else:
                st_start, st_end = allocated_stanzas.get(curr_lyric_s_idx, (3.0, duration_sec - 2.0))
                st_dur = max(1.0, st_end - st_start)

                # Line weights by syllables
                line_syllables = [sum(cls._estimate_syllables(w) for w in l.split()) for l in stanza]
                tot_syllables = max(1, sum(line_syllables))

                line_t = st_start
                for line, syl in zip(stanza, line_syllables):
                    l_dur = st_dur * (syl / tot_syllables)
                    sung_dur = max(0.4, l_dur - 0.15)
                    l_end = min(st_end, line_t + sung_dur)

                    words = line.split()
                    w_sylls = [cls._estimate_syllables(w) for w in words]
                    tot_w_sylls = max(1, sum(w_sylls))
                    timed_words = []
                    w_t = line_t
                    w_tot_dur = max(0.3, l_end - line_t)
                    for w_text, ws in zip(words, w_sylls):
                        wd = w_tot_dur * (ws / tot_w_sylls)
                        timed_words.append({
                            "word": w_text,
                            "start": round(w_t, 2),
                            "end": round(min(l_end, w_t + wd), 2)
                        })
                        w_t += wd

                    timed_results.append({
                        "text": line,
                        "is_section": False,
                        "start": round(line_t, 2),
                        "end": round(l_end, 2),
                        "words": timed_words
                    })
                    line_t += l_dur
                curr_lyric_s_idx += 1

        # Deconflict section headers
        for i, line in enumerate(timed_results):
            if line["is_section"]:
                next_sung = next((l for l in timed_results[i + 1:] if not l["is_section"]), None)
                prev_sung = next((l for l in reversed(timed_results[:i]) if not l["is_section"]), None)
                if next_sung:
                    if prev_sung:
                        line["start"] = round(prev_sung["end"], 2)
                        line["end"] = round(next_sung["start"], 2)
                    else:
                        line["start"] = 0.0
                        line["end"] = round(next_sung["start"], 2)
                else:
                    line["start"] = prev_sung["end"] if prev_sung else 0.0
                    line["end"] = round(min(duration_sec, line["start"] + 1.5), 2)

        timed_results.sort(key=lambda x: x["start"])
        return timed_results

    @classmethod
    def _align_heuristic(cls, lyrics: str, duration_sec: float) -> List[Dict[str, Any]]:
        """
        Tier 3: Syllable-weighted proportional fallback (offline / synthetic testing).
        """
        raw_lines = [l.strip() for l in lyrics.split("\n") if l.strip()]
        if not raw_lines:
            return []

        stanzas: List[List[str]] = []
        current_stanza: List[str] = []
        for line in raw_lines:
            if cls.is_section_header(line):
                if current_stanza:
                    stanzas.append(current_stanza)
                    current_stanza = []
                stanzas.append([line])
            else:
                current_stanza.append(line)
        if current_stanza:
            stanzas.append(current_stanza)

        has_intro = any(cls.is_section_header(s[0]) and "intro" in s[0].lower() for s in stanzas)
        first_start = min(duration_sec * 0.20, 6.0 if has_intro else 2.5)
        last_end = max(first_start + 4.0, duration_sec - 2.5)
        span = max(1.0, last_end - first_start)

        sung_lines = []
        for s in stanzas:
            for l in s:
                if not cls.is_section_header(l):
                    sung_lines.append(l)

        if not sung_lines:
            step = duration_sec / max(1, len(raw_lines))
            return [{
                "text": line,
                "start": round(i * step, 2),
                "end": round((i + 1) * step, 2),
                "is_section": True,
                "words": []
            } for i, line in enumerate(raw_lines)]

        weights = [max(2, sum(cls._estimate_syllables(w) for w in l.split())) for l in sung_lines]
        total_w = sum(weights)

        timed_results: List[Dict[str, Any]] = []
        sung_idx = 0
        cur_t = first_start

        for s in stanzas:
            if len(s) == 1 and cls.is_section_header(s[0]):
                timed_results.append({
                    "text": s[0],
                    "is_section": True,
                    "start": 0.0,
                    "end": 0.0,
                    "words": []
                })
            else:
                for line in s:
                    w = weights[sung_idx]
                    dur = span * (w / total_w)
                    l_start = cur_t
                    l_end = min(last_end, l_start + max(0.5, dur - 0.15))

                    words = line.split()
                    w_syl = [cls._estimate_syllables(wrd) for wrd in words]
                    tot_syl = max(1, sum(w_syl))
                    tw = []
                    wt = l_start
                    for wrd, syl in zip(words, w_syl):
                        wd = (l_end - l_start) * (syl / tot_syl)
                        tw.append({"word": wrd, "start": round(wt, 2), "end": round(min(l_end, wt + wd), 2)})
                        wt += wd

                    timed_results.append({
                        "text": line,
                        "is_section": False,
                        "start": round(l_start, 2),
                        "end": round(l_end, 2),
                        "words": tw
                    })
                    cur_t += dur
                    sung_idx += 1

        for i, line in enumerate(timed_results):
            if line["is_section"]:
                next_sung = next((l for l in timed_results[i + 1:] if not l["is_section"]), None)
                prev_sung = next((l for l in reversed(timed_results[:i]) if not l["is_section"]), None)
                if next_sung:
                    if prev_sung:
                        line["start"] = round(prev_sung["end"], 2)
                        line["end"] = round(next_sung["start"], 2)
                    else:
                        line["start"] = 0.0
                        line["end"] = round(next_sung["start"], 2)
                else:
                    line["start"] = prev_sung["end"] if prev_sung else 0.0
                    line["end"] = round(min(duration_sec, line["start"] + 1.5), 2)

        timed_results.sort(key=lambda x: x["start"])
        return timed_results

    @classmethod
    def align_lyrics(
        cls,
        lyrics: str,
        duration_sec: float,
        vocal_stem_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Main entry point for lyric alignment.
        Executes resilient 3-tier strategy:
          Tier 1: TorchAudio MMS_FA Neural Forced Alignment on vocal stem.
          Tier 2: Multi-interval Adaptive Acoustic VAD.
          Tier 3: Proportional Syllable Estimation.
        """
        if not lyrics or duration_sec <= 0:
            return []

        clean_lyrics = lyrics.strip()
        if not clean_lyrics:
            return []

        resolved_vocal_path = _resolve_audio_file(vocal_stem_path)

        # Tier 1: Neural Forced Alignment (TorchAudio MMS_FA)
        if HAS_TORCH_AUDIO and resolved_vocal_path:
            try:
                results = cls._align_neural_mms_fa(clean_lyrics, duration_sec, resolved_vocal_path)
                if results and any(not r.get("is_section") and r.get("words") for r in results):
                    logger.info(f"Acoustic alignment succeeded via TorchAudio MMS_FA for {resolved_vocal_path}")
                    return results
            except Exception as e:
                logger.warning(f"Neural MMS_FA forced alignment failed, falling back to Tier 2: {e}")

        # Tier 2: Adaptive Acoustic VAD
        if resolved_vocal_path:
            try:
                vocal_regions = cls._extract_vocal_regions(resolved_vocal_path, duration_sec)
                if vocal_regions:
                    logger.info(f"Acoustic alignment using Tier 2 VAD ({len(vocal_regions)} vocal regions)")
                    return cls._align_acoustic_vad(clean_lyrics, duration_sec, vocal_regions)
            except Exception as e:
                logger.warning(f"Acoustic VAD alignment failed, falling back to Tier 3: {e}")

        # Tier 3: Proportional Syllable Heuristic
        logger.info("Using Tier 3 proportional syllable alignment fallback.")
        return cls._align_heuristic(clean_lyrics, duration_sec)

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
            # In LRC, section headers can be included as informational cues
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

