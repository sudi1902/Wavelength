# Sound Effect Extractor & Library — Implementation Plan

**Status: awaiting approval — no code until this plan is signed off.**

A macOS app that ingests short-form social videos (TikTok / Reels), strips out speech and
music, detects the individual sound effects that remain, and files each one as a clean,
labeled `.wav` in a permanent, searchable personal library.

---

## 1. Architecture & Tech Stack

### 1.1 The core decision: which separation model

The task is **not** classic music stem separation (vocals/drums/bass/other) — it is
**Cinematic Audio Source Separation (CASS)**: splitting a mix into *dialogue / music /
effects*. That third stem is exactly what we want to keep. Options evaluated:

| Option | Stems | Fit | Notes |
|---|---|---|---|
| **Bandit-v2** (recommended) | speech / music / **effects** | ✅ Exact fit | Trained on DnR-v3 (the reference CASS dataset), Apache-2.0, PyTorch, checkpoints on Zenodo. Bandit won its category in the Sound Demixing Challenge 2023 cinematic track lineage. |
| Demucs `htdemucs_ft` | vocals / drums / bass / other | ⚠️ Wrong stem semantics | Best-in-class for *music*, mature tooling, fast on Apple Silicon. But "other" = melodic instruments **+** SFX mixed together — background music with synths/guitars would bleed into our keep-stem. |
| BS/Mel-RoFormer family | vocals / instrumental | ❌ | Two-stem karaoke models; no effects stem. |
| MRX (Mitsubishi) | dialogue / music / effects | ⚠️ | Right task, but weaker results than Bandit and less accessible code/weights. |

**Choice: Bandit-v2 as the primary engine, behind a `Separator` interface, with Demucs
`htdemucs` as a fallback engine.**

Reasoning:
- Bandit-v2 is the only well-validated open model whose output stems match the product
  requirement one-to-one (speech→discard, music→discard, effects→keep).
- Demucs would force us to approximate "effects" from stems designed for songs, which is
  the main source of quality risk in this whole project. Better to use the model trained
  for exactly this.
- Risk hedge: Bandit-v2 is research-grade code (the author says so in the README). We
  wrap it behind a thin `Separator` interface (`separate(wav) -> {speech, music, effects}`)
  so we can swap engines without touching the rest of the pipeline. Demucs
  (vocals-removal only, effects ≈ instrumental with a music-detection gate) ships as the
  `--engine demucs` fallback if Bandit setup proves painful on some machine.
- Known limitation to design around: CASS models trained on read-speech route **nonverbal
  vocalizations (laughter, screams) into the effects stem** (this is a documented gap that
  the DnR-nonverbal dataset work addresses). Our labeling stage (CLAP) will down-rank/flag
  segments classified as laughter/speech so they land in a review bucket instead of the
  library.

### 1.2 Speed on Apple Silicon

- Inputs are 15–90 s clips, so even 1× realtime is acceptable; anything faster is gravy.
- PyTorch **MPS** backend gives roughly ~5× speedup over CPU for Demucs-class models;
  MLX ports of htdemucs hit 34–73× realtime on recent Apple Silicon. Bandit-v2 runs on
  PyTorch — we run it with `mps` where its ops are supported, CPU fallback otherwise
  (`PYTORCH_ENABLE_MPS_FALLBACK=1`). For a 60 s clip, worst case (pure CPU on an M1) is
  on the order of a minute; typical will be seconds.
- Verdict: **comfortably fast enough locally**; no cloud needed.

### 1.3 Auto-labeling model

**CLAP (zero-shot, via `laion_clap` or Microsoft `msclap`)** against a *curated SFX
vocabulary* (~60–100 prompts: whoosh, pop, ding, boom, swoosh, click, riser, glitch,
sparkle, camera shutter, notification, record scratch, …).

- CLAP scores each segment against text prompts ("the sound of a whoosh") — 82–91 %
  top-1 on ESC-50, and crucially the vocabulary is *ours to tune*, so labels stay short
  and CapCut-friendly instead of AudioSet's 527-class ontology.
- PANNs (CNN14/AudioSet) is the fallback/alternative — strong tagger, but fixed labels.
- CLAP double-duty: the same scores give us a **junk/bleed filter** (segments scoring
  highest on "music" / "speech" / "silence, noise" get quarantined, not stored).

### 1.4 Language & runtime

**Python 3.11+** for the entire pipeline (the ML models force this; everything else
follows). `ffmpeg` for all demux/transcode. `librosa`/`numpy`/`soundfile` for DSP.
`watchdog` (FSEvents) for folder watching. `osxphotos` for Photos-library ingestion.
SQLite for the library index.

### 1.5 Interface recommendation

**Phase 1–2: CLI. Phase 3: a local web GUI (FastAPI backend + single-page frontend),
launched via a tiny menu-bar wrapper.** Reasoning against the alternatives:

- **SwiftUI native**: best macOS feel, but the pipeline is unavoidably Python — we'd be
  maintaining a two-language app with an IPC bridge for a single-user tool. High cost,
  low benefit.
- **Electron**: ~200 MB runtime to render one library view. Overkill.
- **Tauri**: nice, but bundling a Python sidecar with PyTorch weights into a signed
  `.app` is real packaging pain for v1.
- **Local web app**: the browser is the best free audio-preview widget there is
  (`<audio>` with waveform via wavesurfer.js, instant scrubbing, keyboard-driven
  preview), drag-and-drop of video files onto the page works natively, search/tag UI is
  trivial, and it runs in the *same process* as the watcher and pipeline. `sfx serve`
  opens `http://localhost:8317`. If it ever needs to feel more native, wrapping this
  exact UI in Tauri later is a additive step, not a rewrite.

---

## 2. Pipeline Stages

```
ingest → extract-audio → separate → segment → clean → label → store
```

1. **Ingest** — accept a video path (CLI arg, drag-drop, watcher event, or Photos
   export). Validate with `ffprobe`: is it a media file, does it have an audio stream,
   duration, codec. Compute SHA-256 of the file → skip if this exact video was already
   processed (idempotent re-runs).
2. **Extract audio** — `ffmpeg -i in.mp4 -vn -ac 2 -ar 44100 …` → temp WAV. Handles
   .mp4/.mov/.webm and anything else ffmpeg reads (so HEVC iPhone footage is free).
3. **Separate** — `Separator.separate()` → speech/music/effects stems (44.1 kHz WAV).
   Keep `effects.wav`; optionally archive it per-source for re-segmentation later.
4. **Segment** — on the effects stem: RMS energy gating to find active regions +
   `librosa` spectral-flux onset detection to split multi-hit regions; merge events
   closer than ~150 ms; hard floor/ceiling on segment count sanity.
5. **Clean** — per segment: trim to energy envelope, add 50 ms pre / 100 ms post
   padding, 10 ms fade-in / 30 ms fade-out (click-free), peak-normalize to −1 dBFS
   (optional loudness normalize to −18 LUFS). **Junk filters**: duration < 100 ms,
   peak < −40 dBFS, near-silent RMS, spectral-flatness ≈ 1 (pure hiss) → discard.
6. **Label** — CLAP zero-shot over the SFX vocabulary → top label + confidence.
   Segments whose best match is speech/music/noise → quarantine bucket (reviewable in
   the UI, not auto-deleted). Low-confidence → label `"unknown"`.
7. **Store** — write WAV (16-bit/44.1 kHz PCM; source is 48 kHz → keep 48 kHz, never
   upsample) + optional MP3 (320 kbps preview/export), insert metadata row in SQLite,
   dedupe by audio content hash (same effect from a re-downloaded video isn't stored
   twice).

Each stage is a pure function on files + a context object; stages log structured events
so the UI can show per-video progress.

---

## 3. File / Folder Structure

### Codebase

```
Wavelength/
├── PLAN.md
├── pyproject.toml              # single package, `uv`/pip installable, entry point `sfx`
├── README.md
├── src/sfx/
│   ├── cli.py                  # `sfx extract <video>`, `sfx watch`, `sfx serve`, `sfx library …`
│   ├── config.py               # thresholds, paths, engine choice (TOML at ~/.config/sfx/)
│   ├── pipeline/
│   │   ├── ingest.py
│   │   ├── separate/
│   │   │   ├── base.py         # Separator interface
│   │   │   ├── bandit.py       # primary engine
│   │   │   └── demucs.py       # fallback engine
│   │   ├── segment.py
│   │   ├── clean.py
│   │   ├── label.py            # CLAP + vocabulary.py
│   │   └── store.py
│   ├── library/
│   │   ├── db.py               # SQLite schema + queries
│   │   └── models.py
│   ├── watch.py                # watchdog Downloads watcher
│   ├── photos.py               # osxphotos ingestion
│   └── server/                 # Phase 3: FastAPI + static/ frontend
├── tests/                      # unit tests + fixture audio clips
└── models/                     # downloaded weights (gitignored)
```

### Sound library — **flat files + SQLite index** (not per-video folders)

```
~/SoundEffectsLibrary/
├── library.db                  # all metadata; the folder stays valid if db is rebuilt
├── effects/
│   └── 2026/07/
│       └── 20260708-a3f2__whoosh-01.wav
├── previews/                   #  mp3 mirrors (generated on demand)
├── quarantine/                 # filtered-out-but-maybe segments, purged after N days
└── sources/                    # optional: original videos + full effects stems
```

Why flat + DB rather than per-source-video subfolders: renames/tags/search live in the
database so **file paths never need to change** (a DAW or CapCut project referencing a
file keeps working after you rename or retag it in the library); grouping by source
video is just a query; dedup is a hash lookup. Filenames stay human-readable
(`date-shortid__label-nn.wav`) so the folder is still browsable in Finder without the
app.

**Metadata per effect** (SQLite row): id, filename, content-hash, source video (path +
hash + original filename), extraction datetime, duration, sample rate, peak/LUFS, auto
label + confidence, user label (rename), tags, favorite flag, times-previewed.

---

## 4. Dependencies & System Requirements

| Requirement | Detail |
|---|---|
| macOS | 13+ (Apple Silicon strongly recommended; Intel works, slower) |
| ffmpeg | via Homebrew (`brew install ffmpeg`) — demux, resample, MP3 encode |
| Python | 3.11+, project managed with `uv` |
| PyTorch | ≥ 2.3 with MPS; `PYTORCH_ENABLE_MPS_FALLBACK=1` for unsupported ops |
| Model weights | Bandit-v2 checkpoint (Zenodo, ~0.5–2 GB) + CLAP checkpoint (~0.6 GB) + optional htdemucs (~1 GB) → **~4 GB disk**, downloaded on first run with checksum verification |
| RAM | 8 GB min, 16 GB comfortable |
| Disk | Library grows ~1–10 MB per effect (WAV) |
| Permissions | Full Disk Access or Photos access (TCC prompt) only if Photos ingestion is used; Downloads access prompt for watch mode |

Python deps: `torch`, `librosa`, `soundfile`, `numpy`, `laion_clap` (or `msclap`),
`watchdog`, `osxphotos`, `fastapi`+`uvicorn` (Phase 3), `typer` (CLI), `pydantic`.

---

## 5. Failure Handling

| Failure | Behavior |
|---|---|
| Video has no audio stream | Detected at ffprobe stage → skipped, logged, surfaced in UI ("no audio") |
| No sound effects found (talking-head / music-only video) | Normal outcome, not an error: recorded in DB as processed-with-0-effects so it isn't re-processed; effects stem optionally kept in `sources/` for manual inspection |
| Separation artifacts (music/speech bleed into effects stem) | CLAP post-filter quarantines segments that classify as music/speech; conservative thresholds — quarantine, never silently delete |
| Nonverbal vocals (laughs, screams) leaking into effects stem | Known CASS-model gap; CLAP labels catch most → quarantine with label shown |
| Corrupted / truncated / mislabeled files | ffprobe validation up front; pipeline stage exceptions mark the video `failed` with the error message, never crash the watcher loop |
| Very long videos (someone drops a 2 h screen recording) | Default cap 10 min (configurable, `--force` to override); separation runs in overlapping chunks (~30 s, 1 s crossfade) so memory stays bounded regardless |
| Duplicate ingestion (re-downloaded same TikTok) | Video hash short-circuits; per-effect audio hash prevents dupes even from different encodes |
| Model download failure / bad weights | Checksummed download with resume + clear error message; app degrades to explaining what's missing rather than stack-tracing |
| Watcher races (file still downloading) | Wait for file size to stabilize (2 s quiet period) before ingesting |

---

## 6. Phased Build Order

### Phase 1 — MVP pipeline (CLI)
`sfx extract video.mp4` → separated, segmented, cleaned `.wav` files in `~/SoundEffectsLibrary/effects/`.
- Project scaffolding, config, ffprobe/ffmpeg ingest
- Bandit-v2 engine (weights auto-download) behind `Separator` interface
- Basic segmentation (RMS gating + onsets), basic cleaning (trim/pad/fade), junk filters
- WAV export + minimal SQLite record (source, date, duration)
- **Exit criteria:** drop in a real TikTok with 3–4 obvious SFX → get 3–4 clean, separately-usable WAVs

### Phase 2 — Quality, labeling, metadata
- Segmentation tuning on a corpus of real clips (merge windows, thresholds as config)
- CLAP labeling + SFX vocabulary + music/speech quarantine filter
- Full metadata schema, MP3 export, loudness normalization option, dedup hashing
- Demucs fallback engine; chunked processing for long inputs
- `sfx library list/search/rename/tag/delete/export` CLI commands
- **Exit criteria:** effects come out named ("whoosh-01.wav"), junk rate low enough that you rarely delete manually

### Phase 3 — Library UI + automation
- FastAPI server + web UI: browse/search/filter, waveform preview with instant play,
  rename, tag, favorite, delete, drag-out/export, quarantine review
- Watch-folder mode (`sfx watch ~/Downloads`) with the stabilization logic; toggle in UI
- Photos-library ingestion (`sfx photos --recent 20` / album picker) via osxphotos
- Menu-bar launcher (`sfx serve` autostart) — optional nicety
- **Exit criteria:** AirDrop a video from the phone → effects appear in the library UI, previewable, within a minute, no terminal touched

---

## 7. Open Questions

1. **Your hardware** — which Mac (chip + RAM)? Determines whether we tune for MPS from day one.
2. **Effects inside music** — when the SFX is *baked into the music track* (very common
   in Reels where the "pop" is part of the song), separation will file it under music.
   Acceptable to lose those in v1?
3. **Quarantine review** — happy with "uncertain segments go to a review bucket you can
   promote/delete," or do you prefer aggressive auto-discard (cleaner library, some loss)?
4. **Source archiving** — keep a copy of each original video + its full effects stem in
   `sources/` (enables re-processing with better future models, costs disk), or discard after extraction?
5. **Sample rate policy** — keep native 48 kHz for iPhone-sourced audio (proposed), or force everything to 44.1 kHz for uniformity?
6. **GUI confirmation** — OK with the local-web-UI approach for Phase 3, or do you feel strongly about a fully native `.app` from the start?
7. **Repo naming** — the repo is `Wavelength`; use that as the product/CLI name (`wavelength extract …`) or keep the neutral `sfx` CLI name?

---

## References

- [Bandit-v2 — GitHub (kwatcharasupat/bandit-v2)](https://github.com/kwatcharasupat/bandit-v2) — CASS model, Apache-2.0, DnR-v3
- [Remastering Divide and Remaster (DnR-v3) — IEEE](https://ieeexplore.ieee.org/document/10704085/)
- [The Sound Demixing Challenge 2023 — Cinematic Track](https://arxiv.org/pdf/2308.06981)
- [DnR-nonverbal: nonverbal sounds gap in CASS](https://arxiv.org/pdf/2506.02499)
- [A Generalized Bandsplit Neural Network for CASS (Bandit)](https://www.researchgate.net/publication/376247094_A_Generalized_Bandsplit_Neural_Network_for_Cinematic_Audio_Source_separation)
- [demucs-mlx — Apple Silicon MLX port](https://github.com/ssmall256/demucs-mlx/) and [MLX port writeup (~34× realtime on M4 Max)](https://medium.com/@andradeolivier/i-ported-demucs-to-apple-silicon-it-separates-a-7-minute-song-in-12-seconds-6c4e5cffb5c3)
- [Demucs GUI usage notes (MPS ≈ 5× speedup)](https://github.com/CarlGao4/Demucs-Gui/blob/main/usage.md)
- [CLAP: Learning Audio Concepts From Natural Language Supervision](https://www.researchgate.net/publication/361253229_CLAP_Learning_Audio_Concepts_From_Natural_Language_Supervision)
- [Prompt templates for zero-shot audio classification](https://arxiv.org/pdf/2409.13676)
