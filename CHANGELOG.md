# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0]

### Added

- **`slicer.py`** -- general audio slicing services using cysox, shared
  between Morphagene and Octatrack workflows.
- Slice point detection (returns times in seconds for Morphagene markers):
  `slice_points_by_onsets()` (transient detection via `cysox.onset`),
  `slice_points_by_bpm()` (beat-aligned positions),
  `slice_points_by_count()` (equal division).
- File slicing (writes individual WAVs for Octatrack chaining):
  `slice_file_by_onsets()`, `slice_file_by_bpm()`,
  `slice_file_by_count()` (via `cysox.slice_loop`), and
  `split_file_by_silence()` (via `cysox.split_by_silence`).
- **`mg_utils slice`** subcommand: create Morphagene reels with markers
  from BPM (`--bpm`), equal division (`-n`/`--count`), or onset detection
  (default). Supports `--beats` for beats-per-slice and `-s`/`--splices`
  to limit marker count.
- **`ot_utils slice`** subcommand: slice a single audio file and build
  an Octatrack sample chain. Supports BPM (`--bpm`), count (`-n`),
  onset detection (default), and silence splitting (`--split`). Sliced
  files are auto-chained with `.ot` metadata. Supports `--auto-convert`
  and `--normalize` flags.
- Comprehensive pytest suite for slicer: 25 tests covering slice point
  detection (onsets, BPM, count), file slicing, silence splitting,
  Morphagene integration, and CLI parser extensions for both tools.

- `test_trim_len_uses_tempo_parameter` test to verify trim/loop length
  calculation respects the `tempo` argument.

- **`ot_utils.py`** -- Elektron Octatrack sample chain utility, ported from
  [ot_utils](https://github.com/icaroferre/ot_utils) (Rust) with additional features from [AudioHit](https://github.com/icaroferre/AudioHit).
  Re-implemented in Python with cysox for WAV I/O.

- `ot-utils` console script entry point via `pyproject.toml`.

- `OTSlice` data class representing a single slice in a sample chain
  (start point, length, loop point).

- `read_wav_mono16()` -- reads mono 16-bit WAVs via cysox with format
  validation.

- `write_wav_mono16()` -- writes mono 16-bit PCM WAVs with correct RIFF
  header sizing.

- `generate_ot_data()` -- builds the Octatrack `.ot` binary slice metadata
  (big-endian FORM/DPS1/SMPA header, tempo, trim/loop lengths, up to 64 slice
  entries, checksum).

- `write_ot_file()` -- writes `.ot` sidecar files.

- `build_chain()` -- concatenates WAV files into a sample chain, writes the
  output WAV and `.ot` sidecar. Supports tight packing and evenly-spaced
  (zero-padded) modes.

- `chain` CLI subcommand with `-i`/`--input`, `-o`/`--output`, `--even`,
  `--bpm`, `--sr`, and `-v`/`--verbose` flags.

- **Auto-conversion** (`--auto-convert`): non-conforming inputs (stereo,
  wrong sample rate, wrong bit depth) are automatically converted to mono
  16-bit at the target sample rate via `cysox.convert()`, rather than
  being rejected. Ported from AudioHit's auto-convert behaviour.

- **Peak normalization** (`--normalize`): each sample is peak-normalized
  to -0.1 dBFS before concatenation for consistent volume across slices.
  Ported from AudioHit's pre-chain normalization.

- **Random sample selection** (`--random`, `--seed`): randomly select up
  to 64 samples from a folder, with optional seed for reproducibility.
  Preserves sorted order of the selected subset. Ported from AudioHit's
  `--ot_random` flag.

- `needs_conversion()` and `auto_convert()` utility functions.

- `normalize_samples()` -- integer-domain peak normalization for int16 arrays.

- `select_random()` -- deterministic random subset selection with order preservation.

- Comprehensive pytest suite: 60 tests covering OTSlice, WAV I/O, `.ot`
  binary format, auto-conversion, normalization, random selection, chain
  building (tight/even/auto-convert/normalize/random), and CLI argument
  parsing.

- `resample_file()` -- file-to-file resampling via libsox's polyphase
  resampler (used by `make_reel`).

- `--threshold` / `-t` flag for onset detection sensitivity (0.0-1.0).

- `--sensitivity` flag for onset detection strictness (1.0-3.0).

- `--method` / `-m` flag to select onset algorithm: hfc, flux, energy, complex.

- `TestDetectOnsets` test class exercising cysox onset detection on generated
  audio fixtures.

- `TestResampleFile` test class verifying libsox resampling output.

- Float32 precision preservation test (`test_float32_precision_preserved`).

- CLI parser tests for the new onset flags and their defaults.

- `mg_utils.py` -- unified CLI tool merging the functionality of the former
  `morphagene_ableton.py` and `morphagene_onset.py` into a single argparse-based
  entry point with `ableton` and `onset` subcommands.

- `mg-utils` console script entry point via `pyproject.toml`.

- WAV reader supporting int16, int32, float32, float64, and uint8 inputs.

- WAV writer producing 32-bit float files with correct RIFF header sizing,
  `cue` marker chunks, and `LIST/adtl/labl` label chunks.

- Linear-interpolation resampler for converting arbitrary sample rates to
  the Morphagene's required 48 kHz.

- Ableton Live `.als` parser extracting BPM and locator positions from
  gzipped XML.

- `select_markers()` for evenly thinning a large onset set down to a target
  splice count.

- Morphagene limit warnings (300 splices, 2.9 minutes).

- Comprehensive pytest suite covering I/O, resampling, marker parsing,
  selection, integration pipeline, and CLI argument parsing.

- `Makefile` with `test` target.

### Changed

- Moved `mg_utils.py` to `src/wavtools/morphagene.py` (src layout).

- Added `src/wavtools/__init__.py`.

- Added `[build-system]` (hatchling) and `[tool.hatch.build.targets.wheel]`
  to `pyproject.toml` for src layout discovery.

- Updated `mg_utils` entry point from `mg_utils:main` to
  `wavtools.morphagene:main`.

- Updated test imports from `mg_utils` to `wavtools.morphagene`.

- Replaced librosa with cysox for onset detection. This eliminates the heavy
  librosa/numba dependency tree and uses cysox's native C-level onset detector
  instead.

- Replaced `scipy.io.wavfile` with `cysox.stream` / `cysox.info` for WAV
  reading. cysox reads any format libsox supports, not just PCM WAV.

- IEEE-float WAVs are now read by parsing the RIFF data chunk directly,
  bypassing both cysox's int32 sample pipeline and the stdlib `wave` module,
  which both quantize float32 values through integer conversion.

- `make_reel` now resamples via `cysox.convert` (libsox polyphase resampler)
  instead of naive `numpy.interp` linear interpolation. This eliminates
  aliasing and high-frequency loss during sample-rate conversion.

- `resample()` (per-channel `numpy.interp`) is retained as a lightweight
  utility for in-memory arrays but is no longer used in the main pipeline.

- Onset subcommand now exposes `--threshold`, `--sensitivity`, and `--method`
  flags corresponding to the cysox onset API (previously hardcoded to librosa
  Superflux parameters).

- Dependencies reduced from numpy + scipy + librosa (optional) to numpy + cysox.

### Removed

- `morphagene_ableton.py` -- replaced by `mg_utils.py ableton`.

- `morphagene_onset.py` -- replaced by `mg_utils.py onset`. This also removes
  the ~350-line vendored copy of `scipy.io.wavfile` that was embedded in that
  file.

- `tests/test_morphagene_ableton.py` -- replaced by `tests/test_mg_utils.py`.

### Fixed

- `generate_ot_data()` now uses the caller-supplied `tempo` parameter for
  trim/loop length calculation instead of the hardcoded `OT_DEFAULT_BPM`
  constant. Previously, passing a non-default BPM to `build_chain()` or
  `generate_ot_data()` would encode the correct tempo field but compute
  trim/loop lengths as if BPM were 124.

- Removed unused imports (`import sys` in `morphagene.py`, `from cysox
  import fx` in `octatrack.py`).

- `detect_onsets()` docstring now lists all five supported methods (was
  missing `superflux`).

- `pyproject.toml` version updated from `0.1.0` to `0.3.0` (was stale).

- `pyproject.toml` description replaced placeholder text.

- README: corrected project title from `audiotools` to `wavtools`, fixed
  CLI command names (`mg-utils` -> `mg_utils`), added Octatrack usage
  section, added `superflux` to onset methods table, corrected resampling
  description from "linear interpolation" to "libsox polyphase resampler".

- CHANGELOG v0.3.0: corrected package/module references from
  `audiotools`/`mg_utils.py` to `wavtools`/`morphagene.py`.

- RIFF header size now accounts for cue/label chunks (was only correct for the
  audio data portion in the old Ableton writer).

- Audio array axis convention is now consistently (channels, samples)
  throughout the codebase; the old onset script mixed (samples, channels) and
  (channels, samples) across different functions.
