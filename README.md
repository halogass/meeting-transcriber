# Meeting Transcriber

Local, private meeting transcription with speaker names, for online calls (Zoom, Google Meet, Teams, …).

It records **your microphone** and **the call audio playing on your computer** as two separate streams,
shows **live captions** while you talk, and when you press Stop produces a **refined transcript** where
your voice is labelled with your name and the remote participants are separated by voice
(Speaker 1, Speaker 2, … which you can rename). Then it can write a **summary with action items**.

- Speech recognition: [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (large-v3-turbo, 5-bit quantized; multilingual, handles mixed Indonesian/English)
- Speaker identification: [pyannote.audio](https://github.com/pyannote/pyannote-audio) (`speaker-diarization-community-1`)
- Summaries: Claude (Anthropic API) or any local [Ollama](https://ollama.com) model
- Everything except Claude summaries runs on your machine. Audio never leaves it.

## Features

- Live captions during the meeting, full re-transcription + diarization after it
- Your mic is always "you"; only the call audio is diarized (more accurate, no echo duplicates)
- Echo removal when you use speakers instead of headphones
- Rename speakers, fix text, reassign a paragraph to another speaker, click a timestamp to play
- Meeting history with full-text search; audio kept for playback
- Export to Word (.docx), Markdown, TXT, SRT (and the audio as .opus)
- Built for long meetings (1-3 h+): audio streams to disk as it is recorded and survives a crash or restart
- Optional auto-recording: starts when a call app opens the microphone, stops when the call ends
- Upload any existing audio/video file to transcribe it with all speakers separated
- Uses two GPUs when present (whisper on one, pyannote on the other); falls back to one GPU or CPU
- Plays nicely on shared machines: models are released after 10 idle minutes (Whisper) or right after use (pyannote, Ollama)

## Requirements

- Linux with PulseAudio or PipeWire (Ubuntu 22.04+ / Debian 12+ tested; for uploads only, any Linux or Docker host)
- Optional NVIDIA GPU (4 GB+ VRAM recommended). CPU-only works, just slower.
- ~6 GB disk (Python packages + models), 8 GB RAM
- A free [Hugging Face](https://huggingface.co) account for the pyannote model

## Install

```bash
git clone https://github.com/halogass/meeting-transcriber.git
cd meeting-transcriber
./install.sh
```

The installer installs missing system packages (asks first), builds whisper.cpp (CUDA if available), downloads
the models, sets up the Python environment with [uv](https://docs.astral.sh/uv/) and adds an app-menu entry.

**One-time step for speaker identification:** create a read token at
<https://huggingface.co/settings/tokens> and accept the model terms at
<https://huggingface.co/pyannote/speaker-diarization-community-1>. Paste the token when the installer asks,
or later in **Settings**.

## Run

```bash
./run.sh --open
```

Opens <http://localhost:8765>. Or start **Meeting Transcriber** from your app menu.

1. Join your call as usual.
2. In the app: type a title, check the mic and call-audio devices, press **Start recording**.
3. Press **Stop** when the meeting ends. Processing runs in the background with a progress bar.
4. Rename the speakers, read or export the transcript, generate the summary.

## Docker

The container suits a server that processes uploaded recordings. Live recording also works on a Linux host if
you share the audio socket (see comments in `docker-compose.yml`).

```bash
cp .env.example .env        # add HF_TOKEN (and ANTHROPIC_API_KEY / MT_PASSWORD if needed)
docker compose up --build                                        # NVIDIA GPU (nvidia-container-toolkit)
docker compose --profile cpu up --build meeting-transcriber-cpu  # CPU only
```

Models are downloaded into `./models` on first start; meetings are stored in `./data`.

## Configuration

Everything is in the **Settings** page and stored in `data/settings.json` (file mode 600, it holds API keys).
Environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `MT_HOST` / `MT_PORT` | `127.0.0.1` / `8765` | Listen address |
| `MT_PASSWORD` | empty | Require a password (HTTP basic auth). **Set it if the app is reachable from other machines.** |
| `HF_TOKEN` | empty | Hugging Face token (also read from `huggingface-cli login`) |
| `ANTHROPIC_API_KEY` | empty | Claude summaries |
| `MT_DATA_DIR` / `MT_MODELS_DIR` | `./data` / `./models` | Storage locations |
| `MT_FORCE_CPU` | empty | Ignore GPUs |

More Whisper models: `scripts/download_models.sh large-v3-turbo small-q5_1 large-v3` then pick them in Settings.

**Speed** (GTX 1660 SUPER, 6 GB): the default `large-v3-turbo-q5_0` transcribes about 9x faster than real time
(a 1-hour meeting in ~7 minutes); the full-precision `large-v3-turbo` gave the same words but was 2.5x slower.

## How it works

```
 mic ──parec──► mic.raw ──► live chunker ──► whisper-server ──► live captions (WebSocket)
call ──parec──► call.raw ─┘   (energy VAD)
                    │ Stop
                    ▼
   mic.wav ──► whisper.cpp (VAD, word timestamps) ──────────────┐
  call.wav ──► whisper.cpp (VAD, word timestamps) ──┐           │
          └──► pyannote diarization (other GPU) ────┴─► align words to speakers
                                                      + echo removal ─► transcript.json ─► summary
```

- `whisper-server` (from whisper.cpp) keeps the model loaded; the app talks to it over localhost HTTP.
- Long recordings are transcribed in ~5-minute windows cut at silences.
- Words are assigned to the pyannote speaker turn they overlap most; a sentence is split when the speaker changes.

## Development

```bash
uv sync
uv run pytest
uv run python -m app      # http://localhost:8765
```

Code map: `app/main.py` (API), `app/live.py` (recording + live captions), `app/pipeline.py` (final pass),
`app/whisper.py`, `app/diarize.py`, `app/merge.py`, `app/summarize.py`, `app/static/` (UI, no build step).

## License

MIT. whisper.cpp is MIT; pyannote.audio is MIT; the pyannote model has its own terms on Hugging Face.
