import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.llm_service import _strip_thinking
from app.services.lyrics_graph import sanitize_lyrics


def test_user_exact_example():
    raw = '''- "fire in my chest" - confidence
- "putting doubts to rest" - overcoming
- "rewriting all the rules" - confidence
- "electric, alive, touching the sky" - euphoria
- "unstoppable tonight" - confidence

Yes, this captures the brief well.

Final output ready.</think>..

[Intro]
Light it up, we're lighting up tonight
Walking through the shadows into the light

[Verse 1]
Got this fire in my chest
Putting all my doubts to rest'''

    cleaned = sanitize_lyrics(raw)
    assert 'fire in my chest" - confidence' not in cleaned
    assert "Final output ready" not in cleaned
    assert "</think>" not in cleaned
    assert cleaned.startswith("[Intro]")
    assert "Light it up, we're lighting up tonight" in cleaned
    assert "[Verse 1]" in cleaned


def test_matched_think_tags():
    raw = '''<think>
We need an energetic intro followed by a punchy chorus.
Rhyme scheme: AABB
Let's make it upbeat.
</think>
[Intro]
Here we go now

[Chorus]
We are flying high'''

    cleaned = sanitize_lyrics(raw)
    assert "energetic intro" not in cleaned
    assert "<think>" not in cleaned
    assert "</think>" not in cleaned
    assert cleaned.startswith("[Intro]")
    assert "We are flying high" in cleaned


def test_unmatched_think_open():
    raw = '''<think>
Brainstorming verses...
[Verse 1]
Step into the neon glow'''

    cleaned = sanitize_lyrics(raw)
    assert "Brainstorming" not in cleaned
    assert "Step into the neon glow" in cleaned


def test_markdown_reasoning_blocks():
    raw = '''### Thinking Process
The user wants a cyberpunk theme.
1. Use futuristic metaphors
2. Fast tempo rhythm

**Analysis:** Focus on neon, chrome, circuits.

[Intro]
Neon wires in the dark

[Verse 1]
Circuit heart beating fast'''

    cleaned = sanitize_lyrics(raw)
    assert "Thinking Process" not in cleaned
    assert "Analysis:" not in cleaned
    assert cleaned.startswith("[Intro]")
    assert "Neon wires in the dark" in cleaned


def test_conversational_preamble_and_fences():
    raw = '''```markdown
Here are your completed song lyrics about love and stars:

[Verse 1]
Looking at the constellation

[Chorus]
Shining brighter than the sun
```'''

    cleaned = sanitize_lyrics(raw)
    assert "```" not in cleaned
    assert "Here are your completed song lyrics" not in cleaned
    assert cleaned.startswith("[Verse 1]")
    assert "Looking at the constellation" in cleaned


def test_clean_lyrics_preservation():
    clean_input = '''[Intro]
Bassline starts

[Verse 1]
Walking down the avenue
Everything is fresh and new

[Chorus]
Take a chance on today
Throw your worries away

[Outro]
Fade out'''

    cleaned = sanitize_lyrics(clean_input)
    assert cleaned.strip() == clean_input.strip()


if __name__ == '__main__':
    print("Running Lyrics Cleaning Unit Tests in project conda env...")
    test_user_exact_example()
    print("✓ test_user_exact_example PASSED")
    test_matched_think_tags()
    print("✓ test_matched_think_tags PASSED")
    test_unmatched_think_open()
    print("✓ test_unmatched_think_open PASSED")
    test_markdown_reasoning_blocks()
    print("✓ test_markdown_reasoning_blocks PASSED")
    test_conversational_preamble_and_fences()
    print("✓ test_conversational_preamble_and_fences PASSED")
    test_clean_lyrics_preservation()
    print("✓ test_clean_lyrics_preservation PASSED")
    print("\n🎉 ALL 6 LYRICS SANITIZATION TESTS PASSED SUCCESSFULLY!")
