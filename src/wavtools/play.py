#!/usr/bin/env python3
"""2hp Play sample preparation utility.

Prepare WAV files for the 2hp Play Eurorack sample player by converting them
to mono 16-bit format, enforcing the 32-file limit, generating an options.txt
configuration file, and stripping macOS metadata that causes compatibility
issues.

The 2hp Play requires:
  - WAV files (.wav or .WAV), mono, 16-bit PCM
  - 44.1 kHz recommended (other rates work but alter playback speed)
  - Up to 32 files, sorted alphabetically on a FAT32 micro SD card
  - An optional ``options.txt`` for module configuration

Usage:
    play_utils prepare -i samples/ -o sdcard/ [--auto-convert] [--normalize]
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path

from .core import auto_convert as auto_convert
from .core import needs_conversion as needs_conversion
from .core import normalize_samples as normalize_samples
from .core import read_wav_mono16 as read_wav_mono16
from .core import write_wav_mono16 as write_wav_mono16

log = logging.getLogger(__name__)

PLAY_SAMPLE_RATE = 44100
PLAY_MAX_FILES = 32
PLAY_NORMALIZE_DB = -0.1

# macOS metadata patterns that cause issues on the 2hp Play
MAC_METADATA_PATTERNS = ("._", ".DS_Store", ".Spotlight-V100", ".Trashes", ".fseventsd")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PlayOptions:
    """User-configurable options written to ``options.txt`` on the SD card.

    Each field maps to a boolean flag (0 or 1) in the options file.
    """

    quantize_pitch: bool = False
    """Quantize the Pitch knob and V/Oct CV input to semitones."""

    add_fades: bool = True
    """Add slight fades to sample start/end to prevent pops and clicks."""

    gated_playback: bool = False
    """Play only while the Trig gate is held high."""

    lock_pitch: bool = False
    """Lock each file to its original pitch based on its sample rate."""

    change_on_loop: bool = False
    """Change the active file when a loop restarts (not only on retrigger)."""


# ---------------------------------------------------------------------------
# Options file I/O
# ---------------------------------------------------------------------------

# Map from dataclass field names to the 2hp Play config keys
_FIELD_TO_KEY: dict[str, str] = {
    "quantize_pitch": "QUANTIZE_PITCH",
    "add_fades": "ADD_FADES",
    "gated_playback": "GATED_PLAYBACK",
    "lock_pitch": "LOCK_PITCH",
    "change_on_loop": "CHANGE_ON_LOOP",
}

_KEY_TO_FIELD: dict[str, str] = {v: k for k, v in _FIELD_TO_KEY.items()}


def write_options(path: Path, options: PlayOptions | None = None) -> None:
    """Write an ``options.txt`` configuration file for the 2hp Play.

    Args:
        path: Destination path (typically ``<sd_root>/options.txt``).
        options: Configuration values.  Defaults are used when *None*.
    """
    if options is None:
        options = PlayOptions()

    lines: list[str] = []
    for f in fields(options):
        key = _FIELD_TO_KEY[f.name]
        value = 1 if getattr(options, f.name) else 0
        lines.append(f"{key}={value}")

    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    log.info("Wrote options: %s", path)


def read_options(path: Path) -> PlayOptions:
    """Read an ``options.txt`` configuration file from the SD card.

    Unrecognised keys are silently ignored.  Missing keys fall back to the
    :class:`PlayOptions` defaults.

    Args:
        path: Path to the options file.

    Returns:
        Parsed options.
    """
    kwargs: dict[str, bool] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, _, raw_value = line.partition("=")
        key = key.strip()
        raw_value = raw_value.strip()
        field_name = _KEY_TO_FIELD.get(key)
        if field_name is None:
            continue
        kwargs[field_name] = raw_value == "1"
    return PlayOptions(**kwargs)


# ---------------------------------------------------------------------------
# macOS metadata stripping
# ---------------------------------------------------------------------------


def is_mac_metadata(path: Path) -> bool:
    """Return True if *path* looks like macOS metadata."""
    name = path.name
    return any(name.startswith(prefix) for prefix in MAC_METADATA_PATTERNS)


def strip_mac_metadata(directory: Path) -> list[Path]:
    """Remove macOS metadata files from a directory tree.

    Returns a list of the paths that were removed.
    """
    removed: list[Path] = []
    for item in sorted(directory.rglob("*")):
        if is_mac_metadata(item):
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                import shutil

                shutil.rmtree(item)
            removed.append(item)
            log.info("Removed macOS metadata: %s", item)
    return removed


# ---------------------------------------------------------------------------
# SD card preparation
# ---------------------------------------------------------------------------


def collect_wav_files(input_dir: Path) -> list[Path]:
    """Collect and alphabetically sort WAV files from a directory.

    Only files with a ``.wav`` or ``.WAV`` extension at the top level are
    included (matching the 2hp Play's requirements).
    """
    wav_files: list[Path] = []
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and p.suffix.lower() == ".wav" and not is_mac_metadata(p):
            wav_files.append(p)
    return wav_files


def prepare_card(
    wav_paths: list[Path],
    output_dir: Path,
    *,
    options: PlayOptions | None = None,
    sample_rate: int = PLAY_SAMPLE_RATE,
    auto_convert_enabled: bool = False,
    normalize: bool = False,
    strip_metadata: bool = True,
    write_options_file: bool = True,
    prefix: str | None = None,
) -> list[Path]:
    """Prepare a set of WAV files for loading onto a 2hp Play SD card.

    This is the main entry point.  It:

    1. Validates the file count (max 32).
    2. Optionally converts non-conforming files to mono 16-bit.
    3. Optionally peak-normalizes each sample.
    4. Copies files into *output_dir* with zero-padded numeric prefixes so
       that alphabetical sort matches the intended knob order.
    5. Writes ``options.txt`` if requested.

    Args:
        wav_paths: Ordered list of input WAV files.
        output_dir: Destination directory (will be created if missing).
        options: Module options to write.  Pass *None* for defaults.
        sample_rate: Target sample rate (default 44100).
        auto_convert_enabled: Convert non-conforming files automatically.
        normalize: Peak-normalize each sample to -0.1 dBFS.
        strip_metadata: Remove macOS metadata from *output_dir* after
            copying (default True).
        write_options_file: Write ``options.txt`` to *output_dir*
            (default True).
        prefix: Optional prefix prepended to each numbered filename.
            For example, prefix="kick" yields ``01_kick.wav``.

    Returns:
        List of output WAV paths written into *output_dir*.

    Raises:
        ValueError: If no files are provided or the count exceeds 32.
    """
    if not wav_paths:
        raise ValueError("No input WAV files provided")
    if len(wav_paths) > PLAY_MAX_FILES:
        raise ValueError(
            f"Too many files ({len(wav_paths)}); 2hp Play supports max {PLAY_MAX_FILES}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    tmp_files: list[Path] = []
    pad_width = len(str(len(wav_paths)))

    try:
        for i, src in enumerate(wav_paths, start=1):
            # Determine output filename
            num = str(i).zfill(pad_width)
            stem = src.stem
            if prefix is not None:
                stem = prefix
            out_name = f"{num}_{stem}.wav"
            out_path = output_dir / out_name

            # Convert if needed
            read_path = src
            if auto_convert_enabled and needs_conversion(src, sample_rate):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                tmp_files.append(tmp_path)
                read_path = auto_convert(src, tmp_path, sample_rate)

            if normalize:
                samples = read_wav_mono16(read_path, sample_rate)
                samples = normalize_samples(samples)
                write_wav_mono16(out_path, samples, sample_rate)
            else:
                # Direct copy (already conforming or just converted)
                import shutil

                shutil.copy2(str(read_path), str(out_path))

            written.append(out_path)
            log.info("Prepared %s -> %s", src.name, out_name)
    finally:
        for tmp_path in tmp_files:
            tmp_path.unlink(missing_ok=True)

    if strip_metadata:
        strip_mac_metadata(output_dir)

    if write_options_file:
        write_options(output_dir / "options.txt", options)

    log.info(
        "Prepared %d files in %s (sample rate: %d Hz)",
        len(written),
        output_dir,
        sample_rate,
    )
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="play_utils",
        description="Prepare WAV files for the 2hp Play Eurorack sample player.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")

    sub = parser.add_subparsers(dest="command", required=True)

    # -- prepare subcommand ------------------------------------------------
    prep = sub.add_parser("prepare", help="prepare WAV files for loading onto an SD card")
    prep.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="input directory containing WAV files",
    )
    prep.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="output directory (will be created if it does not exist)",
    )
    prep.add_argument(
        "--sr",
        type=int,
        default=PLAY_SAMPLE_RATE,
        help=f"target sample rate in Hz (default: {PLAY_SAMPLE_RATE})",
    )
    prep.add_argument(
        "--auto-convert",
        action="store_true",
        help="auto-convert non-conforming files (stereo, wrong rate/depth) to mono 16-bit",
    )
    prep.add_argument(
        "--normalize",
        action="store_true",
        help=f"peak-normalize each sample to {PLAY_NORMALIZE_DB} dBFS",
    )
    prep.add_argument(
        "--no-options",
        action="store_true",
        help="do not write options.txt to the output directory",
    )
    prep.add_argument(
        "--no-strip",
        action="store_true",
        help="do not strip macOS metadata files from the output directory",
    )
    prep.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="prefix for numbered output filenames (e.g. --prefix kick -> 01_kick.wav)",
    )

    # options flags
    prep.add_argument(
        "--quantize-pitch",
        action="store_true",
        help="enable QUANTIZE_PITCH in options.txt",
    )
    prep.add_argument(
        "--no-fades",
        action="store_true",
        help="disable ADD_FADES in options.txt (enabled by default)",
    )
    prep.add_argument(
        "--gated",
        action="store_true",
        help="enable GATED_PLAYBACK in options.txt",
    )
    prep.add_argument(
        "--lock-pitch",
        action="store_true",
        help="enable LOCK_PITCH in options.txt",
    )
    prep.add_argument(
        "--change-on-loop",
        action="store_true",
        help="enable CHANGE_ON_LOOP in options.txt",
    )
    prep.set_defaults(func=_cmd_prepare)

    # -- options subcommand ------------------------------------------------
    opt = sub.add_parser("options", help="generate an options.txt file")
    opt.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="output path for options.txt",
    )
    opt.add_argument("--quantize-pitch", action="store_true")
    opt.add_argument("--no-fades", action="store_true")
    opt.add_argument("--gated", action="store_true")
    opt.add_argument("--lock-pitch", action="store_true")
    opt.add_argument("--change-on-loop", action="store_true")
    opt.set_defaults(func=_cmd_options)

    return parser


def _options_from_args(args: argparse.Namespace) -> PlayOptions:
    """Build a PlayOptions from CLI flags."""
    return PlayOptions(
        quantize_pitch=args.quantize_pitch,
        add_fades=not args.no_fades,
        gated_playback=args.gated,
        lock_pitch=args.lock_pitch,
        change_on_loop=args.change_on_loop,
    )


def _cmd_prepare(args: argparse.Namespace) -> None:
    input_dir: Path = args.input
    if not input_dir.is_dir():
        raise SystemExit(f"Error: {input_dir} is not a directory")

    wav_files = collect_wav_files(input_dir)
    if not wav_files:
        raise SystemExit(f"Error: no .wav files found in {input_dir}")

    options = _options_from_args(args) if not args.no_options else None

    prepare_card(
        wav_files,
        args.output,
        options=options,
        sample_rate=args.sr,
        auto_convert_enabled=args.auto_convert,
        normalize=args.normalize,
        strip_metadata=not args.no_strip,
        write_options_file=not args.no_options,
        prefix=args.prefix,
    )


def _cmd_options(args: argparse.Namespace) -> None:
    options = _options_from_args(args)
    write_options(args.output, options)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    args.func(args)


if __name__ == "__main__":
    main()
