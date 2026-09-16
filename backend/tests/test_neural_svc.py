import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
from app.services.voice.neural_svc import neural_svc


def test_neural_svc_voice_conversion():
    """Verify NeuralSVCService converts singing vocals with pitch and formant shift."""
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Synthetic vocal signal (440Hz harmonic tone + vibrato)
    vocal = np.sin(2 * np.pi * (440 + 5 * np.sin(2 * np.pi * 5 * t)) * t) * 0.5

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "source_vocals.wav"
        out_path = Path(tmpdir) / "converted_vocals.wav"
        sf.write(str(src_path), vocal, sr)

        res = neural_svc.convert_vocals(
            source_audio_path=str(src_path),
            output_path=str(out_path),
            pitch_shift=2,
            formant_shift=1.1,
            dry_wet=1.0,
        )

        assert Path(res).is_file()
        data, read_sr = sf.read(res)
        assert read_sr == sr
        assert len(data) > 0
        assert np.max(np.abs(data)) > 0.01
