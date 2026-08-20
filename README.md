# Milimo Music

An open-source AI music generation, neural transcription, and multitrack production platform created by [Mainza Kangombe](https://www.linkedin.com/in/mainza-kangombe-6214295).

Milimo Music pairs state-of-the-art music generation models with digital audio workstation (DAW) editing, note-level transcription (MIDI and MusicXML), stem separation, and offline voice cloning.

---

## Key Capabilities

### Generative Audio Engine
- **Pluggable Providers**: **MiniMax Music 3** is the primary engine — structured multi-section captions `[Verse]`, `[Chorus]`, `[Solo]`, lyric + style-tag conditioning. On Apple Silicon it runs **real native MLX weight inference**; on Windows/Linux it auto-detects the platform and falls back gracefully (no crash). **HeartMuLa-3B** + **HeartCodec** remains available as legacy/local, but is **optional** — it is *not* a hard dependency.
- **Self-Healing Producer**: a weak prompt (e.g. *"A smash hit pop song"*) with little or no lyrics is **intelligently enhanced by the real LLM producer** — it expands the concept into a detailed musical direction and writes genuine, structured lyrics via the AI Co-Writer — so real inference always runs well-conditioned and never degrades to a synthetic placeholder.
- **Inpainting & Continuation**: Extend existing tracks or re-generate specific measures with seamless acoustic transitions.
- **Precision Signal & Sampling Controls**: Configurable duration (5s–300s), CFG scale, temperature, top-k/top-p filtering, DiT diffusion steps, and seed locking.

### Multi-Platform by Design
- Runs on **macOS (Apple Silicon)**, **Windows**, and **Linux**.
- Genuine MiniMax Music 3 MLX inference on Apple; graceful CPU/CUDA fallback elsewhere.
- HeartMuLa/Heartlib is HeartMuLa-only and never required to boot; all its imports are lazy/guarded.
- Dynamic per-instrument stem rendering uses only portable numpy + soundfile.

### MuScriptor Neural Transcription & Engraving
- **Multi-Instrument Polyphonic Transcription**: Extracts individual instrument parts (Piano, Bass, Drums, Vocal melody).
- **Multi-Track MIDI (`.mid`)**: Compatible with Logic Pro, Ableton Live, FL Studio, and Pro Tools.
- **W3C MusicXML 3.1 Sheet Music (`.musicxml`)**: Automatic Grand Staff engraving with standard Treble (𝄞) and Bass (𝄢) clefs.
- **Beat & Tempo Tracking**: Downbeat grid alignment and BPM detection via `beat-this`.

### Web Audio DAW Workspace
- **Dual-Engine Stems**: the DAW exposes **two genuine stem sources you can switch between** — **Per-Instrument** (dynamic, one channel per real instrument in the transcription, labeled with its GM program) and **4 Master Stems** (HTDemucs, real neural source-separation of the master into vocals/drums/bass/other). The dynamic per-instrument view is the default.
- **Arrangement Timeline**: per-instrument (or 4-master) lanes with interactive Solo (`S`) / Mute (`M`) matrix, waveform rendering, and measure navigation. Isolated parts sync to the transcription's real BPM.
- **Piano Roll MIDI Editor**: **dynamic auto-fitting range** built from the transcribed notes (no clamping, no dropped pitches ever), interactive note manipulation/add/delete, live polyphonic Web Audio synthesizer (instrument-aware), and MIDI export.
- **Score Notation Viewer**: **accurate SVG grand-staff engraving** — real pitch-to-staff placement, ledger lines, accidentals, duration-correct note heads, measure bar lines from the track's true beat grid — plus zoom, print, and MusicXML export.
- **Multitrack Console Mixer**: Gain staging, stereo pan sliders, GM-program badges on per-instrument channels, animated LED peak meters, and Matchering reference mastering (-14.0 LUFS broadcast target).

### Studio Workflow & AI Co-Writer
- **Project Folders**: Organize sessions into dedicated projects conditioned on BPM, musical key signature, and tags.
- **Multi-Provider LLM Integration**: Connects to OpenCode Go API, local OMLX Apple Silicon server, Ollama, OpenAI, Google Gemini, and DeepSeek.
- **Agentic Lyrics Engine**: Multi-agent orchestration ensuring structured, rhyme-conscious, and musically aligned lyrics.
- **Voice Training Studio**: Offline singing voice conversion (SVC) and vocal timbre fine-tuning with strict user consent enforcement.

---

## Quickstart

### Prerequisites
- **Python 3.10+** (Recommended: Python 3.12 via Conda)
- **Node.js 18+** & npm
- **Hardware**: macOS with Apple Silicon (MPS / MLX) or Linux / Windows with NVIDIA GPU (CUDA)

### 1. Clone & Environment

```bash
git clone --recurse-submodules https://github.com/mainza-ai/milimomusic.git
cd milimomusic

conda create -n milimomusic python=3.12
conda activate milimomusic
```

### 2. Backend Setup

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
*API docs available at `http://localhost:8000/docs`.*

**Optional — real MiniMax Music 3 inference on Apple Silicon (recommended on M-series Macs):**
Install the native MLX runtime to run the actual model weights instead of the fallback:
```bash
# from the repo root, set the model snapshot path used by the provider
echo 'MINIMAX_MODEL_PATH=/path/to/mlx-community/MiniMax-Music3-bf16' >> .env
pip install mlx
pip install "mlx-audio @ git+https://github.com/Blaizzy/mlx-audio.git@784b29e2691a93ca7483147d86f61859dfaa6296"
```
On Windows/Linux no extra step is needed — the provider auto-detects the platform and uses CPU/CUDA/fallback, so the app always runs. HeartMuLa/Heartlib is optional and never required to start.

### 3. Frontend Setup

In a separate terminal:

```bash
cd frontend
npm install
npm run dev
```
*Application available at `http://localhost:5173`.*

---

## License

Open-source and non-commercial. Created by [Mainza Kangombe](https://www.linkedin.com/in/mainza-kangombe-6214295).
