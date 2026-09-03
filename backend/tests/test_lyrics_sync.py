import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.transcription.karaoke import LyricSyncEngine, lyric_sync_engine


def test_section_tag_detection():
    assert lyric_sync_engine.is_section_header('[Intro]') is True
    assert lyric_sync_engine.is_section_header('[Verse 1]') is True
    assert lyric_sync_engine.is_section_header('[Chorus]') is True
    assert lyric_sync_engine.is_section_header('[Guitar Solo]') is True
    assert lyric_sync_engine.is_section_header('(Bridge)') is True
    assert lyric_sync_engine.is_section_header('Just a regular lyric line') is False


def test_acoustic_alignment_timing_monotonicity():
    lyrics = """[Intro]
Welcome to the show

[Verse 1]
Electric lights in the night
Running fast, feeling right

[Chorus]
We are unstoppable now
Reaching up to the clouds

[Outro]
Fade to white"""

    results = lyric_sync_engine.align_lyrics(lyrics, duration_sec=120.0)
    assert len(results) > 0

    # Section headers must have is_section = True
    sections = [r for r in results if r.get('is_section')]
    assert len(sections) == 4
    assert sections[0]['text'] == '[Intro]'

    # Sung lines must have words
    sung = [r for r in results if not r.get('is_section')]
    assert len(sung) == 6

    # Check timestamps monotonicity and non-negative
    for i in range(len(results)):
        assert results[i]['start'] >= 0.0
        assert results[i]['end'] >= results[i]['start']
        if i > 0:
            assert results[i]['start'] >= results[i-1]['start']

    # Check word timings
    for line in sung:
        assert len(line['words']) > 0
        for w in line['words']:
            assert w['start'] <= w['end']


def test_lrc_and_srt_generation():
    lines = [
        {'text': '[Verse 1]', 'start': 4.0, 'end': 6.0, 'is_section': True, 'words': []},
        {'text': 'Hello world', 'start': 6.5, 'end': 9.0, 'is_section': False, 'words': [{'word': 'Hello', 'start': 6.5, 'end': 7.5}, {'word': 'world', 'start': 7.6, 'end': 9.0}]}
    ]
    lrc = lyric_sync_engine.generate_lrc(lines, title='Test Song')
    assert '[ti:Test Song]' in lrc
    assert '[00:06.50]Hello world' in lrc

    srt = lyric_sync_engine.generate_srt(lines)
    assert "1\n00:00:04,000 --> 00:00:06,000\n[Verse 1]" in srt
    assert "2\n00:00:06,500 --> 00:00:09,000\nHello world" in srt


def test_neural_forced_alignment_on_real_vocal_stem():
    vocal_stem = "backend/generated_audio/stems/53e4c875-484d-4e6d-9db2-10cc40e6bb30_vocals.wav"
    if not os.path.exists(vocal_stem):
        return  # Skip if stem file is not present in environment

    lyrics = """[Verse 1]
Black basalt, frozen breath —
solar wind threads the twilight.
Green sparks awake the sky.

[Outro]
Awakening."""

    results = lyric_sync_engine.align_lyrics(lyrics, duration_sec=14.5, vocal_stem_path=vocal_stem)
    assert len(results) > 0

    # Ensure sections and lines are present
    sections = [r for r in results if r.get('is_section')]
    sung = [r for r in results if not r.get('is_section')]
    assert len(sections) == 2
    assert len(sung) == 4

    # Ensure words are acoustically aligned with non-zero spans
    for s in sung:
        assert len(s['words']) > 0
        for w in s['words']:
            assert w['end'] > w['start']

    # Section header deconfliction: section headers must end before or at sung line start
    for i, r in enumerate(results):
        if r.get('is_section') and i + 1 < len(results) and not results[i+1].get('is_section'):
            assert r['end'] <= results[i+1]['start'] + 0.05


def test_audio_path_resolver():
    from app.transcription.karaoke import _resolve_audio_file
    vocal_stem = "backend/generated_audio/stems/53e4c875-484d-4e6d-9db2-10cc40e6bb30_vocals.wav"
    if os.path.exists(vocal_stem):
        # Test virtual audio prefix resolution
        resolved = _resolve_audio_file("/audio/stems/53e4c875-484d-4e6d-9db2-10cc40e6bb30_vocals.wav")
        assert resolved is not None
        assert os.path.exists(resolved)


if __name__ == '__main__':
    test_section_tag_detection()
    test_acoustic_alignment_timing_monotonicity()
    test_lrc_and_srt_generation()
    test_neural_forced_alignment_on_real_vocal_stem()
    test_audio_path_resolver()
    print('All LyricSyncEngine unit tests passed successfully!')
