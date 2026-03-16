#!/usr/bin/env python3
"""Sample Drum sample preparation utility.

Prepare WAV files for the Sample Drum module by converting them to mono
16-bit format, organising them into folders under a ``SAMPLES`` directory,
and validating against the device's 32 MB RAM / 5-minute limits.

The Sample Drum requires:
  - WAV files, mono, 16-bit PCM
  - 48 kHz recommended (lower rates work and are interpolated at playback)
  - Total loaded samples must fit in 32 MB RAM (approx. 5 minutes at 48 kHz)
  - Samples organised in folders under a top-level ``SAMPLES`` directory

Stereo files can be loaded by extracting a single channel (L or R).
When processing a folder, the left channel is used automatically.

Usage:
    python -m wavtools.sample_drum prepare -i kit/ -o SAMPLES/mykit
"""

from __future__ import annotations

import argparse
import logging
import shutil
import tempfile
from pathlib import Path

import cysox
import numpy as np

from .core import normalize_samples as normalize_samples
from .core import write_wav_mono16 as write_wav_mono16

log = logging.getLogger(__name__)

SD_SAMPLE_RATE = 48000
SD_MAX_RAM_BYTES = 32 * 1024 * 1024  # 32 MB
SD_MAX_DURATION_SEC = 300  # 5 minutes
SD_NORMALIZE_DB = -0.1


# ---------------------------------------------------------------------------
# Stereo channel extraction
# ---------------------------------------------------------------------------


def extract_channel(
    path: Path,
    output_path: Path,
    channel: str = "L",
    sample_rate: int = SD_SAMPLE_RATE,
) -> Path:
    """Extract a single channel from a stereo WAV and write mono 16-bit.

    Args:
        path: Input WAV file (must be stereo).
        output_path: Destination for the mono output.
        channel: ``"L"`` for left (default) or ``"R"`` for right.
        sample_rate: Target sample rate for the output.

    Returns:
        *output_path*.

    Raises:
        ValueError: If *channel* is not ``"L"`` or ``"R"``, or the file
            is not stereo.
    """
    if channel not in ("L", "R"):
        raise ValueError(f"channel must be 'L' or 'R', got {channel!r}")

    meta = cysox.info(str(path))
    if meta.channels != 2:
        raise ValueError(f"{path.name}: expected stereo (2 channels), got {meta.channels}")

    # Read via cysox, optionally resampling first
    need_resample = meta.sample_rate != sample_rate
    if need_resample:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            cysox.convert(
                str(path),
                str(tmp_path),
                sample_rate=sample_rate,
            )
            read_path = tmp_path
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
    else:
        read_path = path
        tmp_path = None

    try:
        chunks = []
        for chunk in cysox.stream(str(read_path)):
            chunks.append(np.frombuffer(chunk, dtype=np.int32).copy())
        raw = np.concatenate(chunks)

        # cysox streams interleaved stereo: [L0, R0, L1, R1, ...]
        interleaved = (raw >> 16).astype(np.int16)
        mono = interleaved[0::2] if channel == "L" else interleaved[1::2]

        write_wav_mono16(output_path, mono, sample_rate)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    log.info("Extracted %s channel from %s -> %s", channel, path.name, output_path.name)
    return output_path


# ---------------------------------------------------------------------------
# WAV validation and conversion
# ---------------------------------------------------------------------------


def needs_conversion(path: Path, sample_rate: int = SD_SAMPLE_RATE) -> bool:
    """Check whether a WAV needs conversion for Sample Drum compatibility."""
    meta = cysox.info(str(path))
    return bool(meta.channels != 1 or meta.sample_rate != sample_rate or meta.bits_per_sample != 16)


def auto_convert(
    path: Path,
    output_path: Path,
    sample_rate: int = SD_SAMPLE_RATE,
    stereo_channel: str = "L",
) -> Path:
    """Convert an audio file to mono 16-bit at the target sample rate.

    For stereo files, *stereo_channel* selects which channel to extract
    (``"L"`` or ``"R"``).  Mono files that only need sample-rate or
    bit-depth conversion are handled via cysox.

    If the file already conforms, it is returned as-is.
    """
    meta = cysox.info(str(path))

    if meta.channels == 1 and meta.sample_rate == sample_rate and meta.bits_per_sample == 16:
        return path

    if meta.channels == 2:
        extract_channel(path, output_path, channel=stereo_channel, sample_rate=sample_rate)
        return output_path

    # Mono but wrong rate or bit depth
    log.info(
        "Converting %s (%d ch, %d Hz, %d bit) -> mono, %d Hz, 16 bit",
        path.name,
        meta.channels,
        meta.sample_rate,
        meta.bits_per_sample,
        sample_rate,
    )
    cysox.convert(
        str(path),
        str(output_path),
        sample_rate=sample_rate,
        channels=1,
        bits=16,
    )
    return output_path


# ---------------------------------------------------------------------------
# RAM estimation
# ---------------------------------------------------------------------------


def ram_bytes(n_samples: int) -> int:
    """Return the RAM footprint of *n_samples* 16-bit samples in bytes."""
    return n_samples * 2


def estimate_ram_usage(
    paths: list[Path],
    sample_rate: int = SD_SAMPLE_RATE,
) -> int:
    """Estimate total RAM usage in bytes for a list of mono 16-bit WAV files.

    Each file is assumed to be mono 16-bit at *sample_rate*.
    """
    total = 0
    for p in paths:
        meta = cysox.info(str(p))
        total += ram_bytes(meta.samples)
    return total


def estimate_duration(
    paths: list[Path],
    sample_rate: int = SD_SAMPLE_RATE,
) -> float:
    """Estimate total playback duration in seconds."""
    total_samples = 0
    for p in paths:
        meta = cysox.info(str(p))
        total_samples += meta.samples
    return total_samples / sample_rate


# ---------------------------------------------------------------------------
# Folder preparation
# ---------------------------------------------------------------------------


def collect_wav_files(input_dir: Path) -> list[Path]:
    """Collect and alphabetically sort WAV files from a directory."""
    wav_files: list[Path] = []
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and p.suffix.lower() == ".wav":
            wav_files.append(p)
    return wav_files


def prepare_folder(
    wav_paths: list[Path],
    output_dir: Path,
    *,
    sample_rate: int = SD_SAMPLE_RATE,
    auto_convert_enabled: bool = False,
    stereo_channel: str = "L",
    normalize: bool = False,
    validate_ram: bool = True,
) -> list[Path]:
    """Prepare WAV files for a Sample Drum sample folder.

    Files are converted (if needed), optionally normalised, and copied into
    *output_dir*.  The caller is responsible for placing *output_dir* under
    the ``SAMPLES`` root on the SD card.

    Args:
        wav_paths: Ordered list of input WAV files.
        output_dir: Destination folder (created if missing).
        sample_rate: Target sample rate (default 48000).
        auto_convert_enabled: Convert non-conforming files automatically.
            Stereo files have channel *stereo_channel* extracted.
        stereo_channel: ``"L"`` or ``"R"`` -- which channel to keep from
            stereo files (default ``"L"``).
        normalize: Peak-normalize each sample to -0.1 dBFS.
        validate_ram: Raise if total size exceeds 32 MB (default True).

    Returns:
        List of output WAV paths.

    Raises:
        ValueError: If no files provided, or RAM/duration limits exceeded.
    """
    if not wav_paths:
        raise ValueError("No input WAV files provided")

    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    tmp_files: list[Path] = []

    try:
        for src in wav_paths:
            out_path = output_dir / src.name

            read_path = src
            if auto_convert_enabled and needs_conversion(src, sample_rate):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                tmp_files.append(tmp_path)
                read_path = auto_convert(src, tmp_path, sample_rate, stereo_channel)

            if normalize:
                from .core import read_wav_mono16

                samples = read_wav_mono16(read_path, sample_rate)
                samples = normalize_samples(samples)
                write_wav_mono16(out_path, samples, sample_rate)
            else:
                shutil.copy2(str(read_path), str(out_path))

            written.append(out_path)
            log.info("Prepared %s -> %s", src.name, out_path.name)
    finally:
        for tmp_path in tmp_files:
            tmp_path.unlink(missing_ok=True)

    if validate_ram and written:
        total_bytes = estimate_ram_usage(written, sample_rate)
        total_duration = estimate_duration(written, sample_rate)
        if total_bytes > SD_MAX_RAM_BYTES:
            raise ValueError(
                f"Total size {total_bytes / (1024 * 1024):.1f} MB exceeds "
                f"Sample Drum RAM limit of {SD_MAX_RAM_BYTES / (1024 * 1024):.0f} MB"
            )
        if total_duration > SD_MAX_DURATION_SEC:
            raise ValueError(
                f"Total duration {total_duration:.1f}s exceeds "
                f"Sample Drum limit of {SD_MAX_DURATION_SEC}s"
            )

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
        prog="wavtools.sample_drum",
        description="Prepare WAV files for the Sample Drum module.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")

    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="prepare WAV files for a Sample Drum folder")
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
        help="output directory (e.g. SAMPLES/mykit)",
    )
    prep.add_argument(
        "--sr",
        type=int,
        default=SD_SAMPLE_RATE,
        help=f"target sample rate in Hz (default: {SD_SAMPLE_RATE})",
    )
    prep.add_argument(
        "--auto-convert",
        action="store_true",
        help="auto-convert non-conforming files to mono 16-bit",
    )
    prep.add_argument(
        "--channel",
        choices=["L", "R"],
        default="L",
        help="which channel to extract from stereo files (default: L)",
    )
    prep.add_argument(
        "--normalize",
        action="store_true",
        help=f"peak-normalize each sample to {SD_NORMALIZE_DB} dBFS",
    )
    prep.add_argument(
        "--no-validate",
        action="store_true",
        help="skip RAM and duration validation",
    )
    prep.set_defaults(func=_cmd_prepare)

    return parser


def _cmd_prepare(args: argparse.Namespace) -> None:
    input_dir: Path = args.input
    if not input_dir.is_dir():
        raise SystemExit(f"Error: {input_dir} is not a directory")

    wav_files = collect_wav_files(input_dir)
    if not wav_files:
        raise SystemExit(f"Error: no .wav files found in {input_dir}")

    prepare_folder(
        wav_files,
        args.output,
        sample_rate=args.sr,
        auto_convert_enabled=args.auto_convert,
        stereo_channel=args.channel,
        normalize=args.normalize,
        validate_ram=not args.no_validate,
    )


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
