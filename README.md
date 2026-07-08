# Wavelength

Extract and isolate individual sound effects from short-form videos (TikTok,
Instagram Reels) into a permanent, searchable personal library on macOS.

Drop in a video → speech and music are stripped away by a cinematic
source-separation model → each remaining sound effect is detected, trimmed,
faded, and saved as its own high-quality WAV in `~/SoundEffectsLibrary/`.

See [PLAN.md](PLAN.md) for the full architecture and roadmap.

## Requirements

- macOS 13+ (Apple Silicon recommended; works elsewhere too)
- [Homebrew](https://brew.sh) `ffmpeg`: `brew install ffmpeg`
- Python 3.11+

## Install

```bash
git clone <this repo> && cd Wavelength
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
wavelength setup bandit    # one-time: installs the separation model (~500 MB)
```

## Use

```bash
# From a link — TikTok, Instagram, YouTube, and ~1,800 other sites
wavelength extract "https://www.tiktok.com/@user/video/123..."

# Instagram usually needs your logged-in browser session:
wavelength extract "https://www.instagram.com/reel/AbC123.../" --browser chrome

# From a local file
wavelength extract ~/Downloads/funny-video.mp4
#   Probing funny-video.mp4
#   Separating stems (bandit)
#   Found 5 candidate segment(s)
#   + ~/SoundEffectsLibrary/effects/2026/07/20260708-a3f2c1__effect-01.wav
#   ...
#   4 effect(s) saved — 1 rejected (too_quiet)

wavelength library         # list everything extracted so far
wavelength info            # config + engine status
```

## Separation engines

| Engine | What it is | When to use |
|---|---|---|
| `bandit` (default) | BandIt Plus cinematic separation (speech / music / **effects**), DnR test SDR 11.50, via [MSST](https://github.com/ZFTurbo/Music-Source-Separation-Training) in an isolated venv | Always, unless setup fails |
| `demucs` | htdemucs music stems, effects ≈ "other" stem (`pip install -e ".[demucs]"`) | Fallback; melodic music bleeds into effects |
| `none` | No separation — whole mix treated as effects | Videos with no speech/music |

```bash
wavelength extract video.mp4 --engine none
```

## Library layout

```
~/SoundEffectsLibrary/
├── library.db              # all metadata (SQLite)
├── effects/2026/07/        # flat, date-organized 16-bit WAVs
└── sources/<hash>/         # archived originals + full effects stems
```

Files are stored at the separation engine's native rate (44.1 kHz for
bandit/demucs; the source's own rate, min 44.1 kHz, for `none`) as 16-bit
PCM — drag them straight into CapCut, Premiere, or any DAW.

Configuration lives at `~/.config/wavelength/config.toml` (see
`src/wavelength/config.py` for available keys and defaults).

## Development

```bash
pip install -e ".[dev]"
pytest
```
