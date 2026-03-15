# wavtools

Utilities for creating hardware synthesizer/sampler audio formats:

- **Morphagene reels** -- 32-bit float / 48 kHz WAV files with embedded splice
  markers, compatible with the [Make Noise Morphagene](https://www.makenoisemusic.com/modules/morphagene).

- **Octatrack sample chains** -- concatenated 16-bit mono WAV files with `.ot`
  slice metadata, compatible with the [Elektron Octatrack](https://www.elektron.se/products/octatrack-mkii/).

## Requirements

- Python >= 3.13
- numpy, cysox (installed automatically)

## Installation

```sh
uv sync
```

## Morphagene usage

Markers can come from two sources:

- **Ableton Live** locators extracted from `.als` project files
- **Automatic onset detection** via cysox (HFC, spectral flux, energy, complex, or superflux)

### Ableton Live locators

Extract locator positions from an Ableton `.als` project and write them as
splice markers into a Morphagene-ready WAV:

```sh
mg_utils ableton -w input.wav -l project.als -o reel.wav
```

### Automatic onset detection

Detect percussive onsets and use them as splice points:

```sh
mg_utils onset -w input.wav -o reel.wav
```

Limit the number of splices (useful for staying under the Morphagene's
300-splice cap):

```sh
mg_utils onset -w input.wav -o reel.wav -s 50
```

Tune onset detection sensitivity and algorithm:

```sh
# More sensitive detection using spectral flux
mg_utils onset -w input.wav -o reel.wav -t 0.1 -m flux

# Stricter detection for clean recordings
mg_utils onset -w input.wav -o reel.wav --sensitivity 2.5
```

### mg_utils options

```text
mg_utils [-v] {ableton,onset} ...

positional arguments:
  {ableton,onset}
    ableton        markers from Ableton Live locators (.als)
    onset          markers from onset detection (cysox)

options:
  -v, --verbose    verbose output

onset options:
  -t, --threshold    detection threshold 0.0-1.0, lower = more sensitive (default: 0.3)
  --sensitivity      detection strictness 1.0-3.0, higher = stricter (default: 1.5)
  -m, --method       onset algorithm: hfc, flux, energy, complex, superflux (default: hfc)
  -s, --splices      max splice markers (default: all detected onsets)
```

### Onset detection methods

| Method      | Best for                        | Speed   |
|-------------|---------------------------------|---------|
| `hfc`       | Drums / percussive material     | Fast    |
| `flux`      | Mixed material / tonal changes  | Medium  |
| `energy`    | Clean recordings, isolated hits | Fastest |
| `complex`   | Phase-sensitive detection       | Slowest |
| `superflux` | Complex onsets with vibrato     | Medium  |

### What it does

1. Reads the input WAV via cysox (any format libsox supports)
2. Normalises audio to float32 in [-1, 1]
3. Resamples to 48 kHz if needed (libsox polyphase resampler)
4. Converts marker times (seconds) to sample-accurate frame positions
5. Writes a 32-bit float WAV with embedded `cue` and `LIST/adtl/labl` chunks

### Morphagene limits

- Max 300 splice markers per reel
- Max ~2.9 minutes of audio per reel

The tool warns (but does not block) when these limits are exceeded.

## Octatrack usage

Build sample chains from a folder of WAV files:

```sh
ot_utils chain -i samples/ -o chain.wav
```

With evenly-spaced slices (zero-pad shorter samples to match the longest):

```sh
ot_utils chain -i samples/ -o chain.wav --even
```

Auto-convert non-conforming files (stereo, wrong sample rate/bit depth):

```sh
ot_utils chain -i samples/ -o chain.wav --auto-convert
```

Peak-normalize each sample before chaining:

```sh
ot_utils chain -i samples/ -o chain.wav --normalize
```

Randomly select up to 64 samples from a large folder:

```sh
ot_utils chain -i samples/ -o chain.wav --random --seed 42
```

### ot_utils options

```text
ot_utils [-v] chain ...

options:
  -v, --verbose       verbose output

chain options:
  -i, --input         input directory containing WAV files
  -o, --output        output WAV file path (the .ot file is created alongside)
  --even              zero-pad shorter samples to match the longest
  --bpm N             tempo in BPM for the .ot metadata (default: 124)
  --sr N              expected sample rate in Hz (default: 44100)
  --auto-convert      auto-convert non-conforming files to mono 16-bit
  --normalize         peak-normalize each sample to -0.1 dBFS
  --random            randomly select up to 64 samples from input
  --seed N            seed for random selection (for reproducibility)
```

### Octatrack limits

- Max 64 slices per chain

## Development

```sh
uv sync
make test
```

## License

See [LICENSE](LICENSE) for details.
