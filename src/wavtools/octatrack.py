#!/usr/bin/env python3
"""Octatrack sample chain utility.

Concatenate multiple WAV files into a single sample chain and generate
Elektron Octatrack ``.ot`` slice metadata files.

Based on https://github.com/icaroferre/ot_utils and
https://github.com/icaroferre/AudioHit (Rust implementations by Icaro Ferre),
re-implemented in Python with cysox for WAV I/O.

Usage:
    ot_utils chain -i samples/ -o output.wav [--even] [--bpm 124] [--sr 44100]
"""

from __future__ import annotations

import argparse
import logging
import random
import struct
import tempfile
from pathlib import Path

import cysox
import numpy as np

log = logging.getLogger(__name__)

OT_SAMPLE_RATE = 44100
OT_MAX_SLICES = 64
OT_DEFAULT_BPM = 124
OT_NORMALIZE_DB = -0.1


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class OTSlice:
    """Represents a single slice in an Octatrack sample chain."""

    __slots__ = ("length", "loop_point", "start_point")

    def __init__(self, start_point: int, length: int, loop_point: int | None = None):
        self.start_point = start_point
        self.length = length
        self.loop_point = loop_point if loop_point is not None else length


# ---------------------------------------------------------------------------
# WAV I/O (16-bit mono, matching Octatrack requirements)
# ---------------------------------------------------------------------------


def needs_conversion(path: Path, sample_rate: int = OT_SAMPLE_RATE) -> bool:
    """Check whether a WAV file needs conversion for Octatrack compatibility."""
    meta = cysox.info(str(path))
    return bool(meta.channels != 1 or meta.sample_rate != sample_rate or meta.bits_per_sample != 16)


def auto_convert(
    path: Path,
    output_path: Path,
    sample_rate: int = OT_SAMPLE_RATE,
) -> Path:
    """Convert an audio file to mono 16-bit at the target sample rate.

    If the file already conforms, it is returned as-is (output_path is not
    written). Otherwise cysox.convert is used and output_path is returned.
    """
    if not needs_conversion(path, sample_rate):
        return path

    meta = cysox.info(str(path))
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


def normalize_samples(samples: np.ndarray, target_db: float = OT_NORMALIZE_DB) -> np.ndarray:
    """Peak-normalize int16 samples to a target dB level.

    Args:
        samples: 1-D int16 array.
        target_db: Target peak level in dBFS (default: -0.1).

    Returns:
        Normalized int16 array.
    """
    peak = np.max(np.abs(samples.astype(np.float64)))
    if peak == 0:
        return samples
    target_linear = 32767.0 * (10.0 ** (target_db / 20.0))
    gain = target_linear / peak
    normalized = np.clip(samples.astype(np.float64) * gain, -32768, 32767)
    result: np.ndarray = normalized.astype(np.int16)
    return result


def read_wav_mono16(path: Path, sample_rate: int = OT_SAMPLE_RATE) -> np.ndarray:
    """Read a WAV file and return 16-bit mono samples as an int16 array.

    Validates that the file is mono and matches the expected sample rate.
    Uses cysox for reading, converting the int32 stream to int16.
    """
    meta = cysox.info(str(path))

    if meta.channels != 1:
        raise ValueError(f"{path.name}: expected mono (1 channel), got {meta.channels} channels")
    if meta.sample_rate != sample_rate:
        raise ValueError(f"{path.name}: expected {sample_rate} Hz, got {meta.sample_rate} Hz")

    chunks = []
    for chunk in cysox.stream(str(path)):
        chunks.append(np.frombuffer(chunk, dtype=np.int32).copy())
    raw = np.concatenate(chunks)
    # cysox streams 32-bit int; shift down to 16-bit range
    samples = (raw >> 16).astype(np.int16)
    return samples


def write_wav_mono16(
    path: Path,
    samples: np.ndarray,
    sample_rate: int = OT_SAMPLE_RATE,
) -> None:
    """Write a 16-bit mono PCM WAV file."""
    samples = np.ascontiguousarray(samples, dtype=np.int16)
    n_samples = len(samples)
    data_bytes = n_samples * 2

    with open(str(path), "wb") as f:
        # RIFF header -- size placeholder, patched at end
        f.write(struct.pack("<4sI4s", b"RIFF", 0, b"WAVE"))

        # fmt chunk (PCM)
        f.write(
            struct.pack(
                "<4sIHHIIHH",
                b"fmt ",
                16,
                1,
                1,
                sample_rate,
                sample_rate * 2,
                2,
                16,
            )
        )

        # data chunk
        f.write(struct.pack("<4sI", b"data", data_bytes))
        f.write(samples.tobytes())

        # Patch RIFF size
        file_size = f.tell()
        f.seek(4)
        f.write(struct.pack("<I", file_size - 8))


# ---------------------------------------------------------------------------
# .ot file generation
# ---------------------------------------------------------------------------


def _push_u32_be(buf: bytearray, value: int) -> None:
    """Append a 32-bit unsigned integer in big-endian format."""
    buf.extend(struct.pack(">I", value))


def _push_u16_be(buf: bytearray, value: int) -> None:
    """Append a 16-bit unsigned integer in big-endian format."""
    buf.extend(struct.pack(">H", value))


def generate_ot_data(
    slices: list[OTSlice],
    total_samples: int,
    sample_rate: int = OT_SAMPLE_RATE,
    tempo: int = OT_DEFAULT_BPM,
) -> bytes:
    """Build the binary content of an Octatrack .ot file.

    The format is big-endian and follows the reverse-engineered specification
    from the OctaChainer / ot_utils projects.

    Args:
        slices: List of OTSlice objects (max 64).
        total_samples: Total number of samples in the chain WAV.
        sample_rate: Sample rate of the WAV (typically 44100).
        tempo: BPM value embedded in the .ot metadata.

    Returns:
        Raw bytes for the .ot file.
    """
    if len(slices) > OT_MAX_SLICES:
        raise ValueError(f"Octatrack supports max {OT_MAX_SLICES} slices, got {len(slices)}")

    buf = bytearray()

    # Header: FORM....DPS1SMPA + version bytes
    buf.extend(b"FORM")
    buf.extend(b"\x00\x00\x00\x00")  # placeholder for size
    buf.extend(b"DPS1")
    buf.extend(b"SMPA")
    buf.extend(bytes([0x00, 0x00, 0x00]))
    buf.extend(bytes([0x00, 0x02, 0x00]))

    # Tempo: tempo * 6 * 4
    _push_u32_be(buf, tempo * 6 * 4)

    # TrimLen and LoopLen
    trim_len = int((tempo * total_samples / (sample_rate * 60)) + 0.5) * 25
    _push_u32_be(buf, trim_len)  # TrimLen
    _push_u32_be(buf, trim_len)  # LoopLen

    # Stretch
    _push_u32_be(buf, 0)

    # Loop
    _push_u32_be(buf, 0)

    # Gain
    _push_u16_be(buf, 48)

    # Quantize
    buf.append(0xFF)

    # TrimStart
    _push_u32_be(buf, 0)

    # TrimEnd
    _push_u32_be(buf, total_samples)

    # LoopPoint
    _push_u32_be(buf, 0)

    # 64 slice entries (12 bytes each)
    for i in range(OT_MAX_SLICES):
        if i < len(slices):
            s = slices[i]
            _push_u32_be(buf, s.start_point)
            _push_u32_be(buf, s.start_point + s.length)
            _push_u32_be(buf, s.loop_point)
        else:
            buf.extend(bytes(12))

    # Slice count
    _push_u32_be(buf, len(slices))

    # Checksum: sum of all bytes from offset 16 onward
    checksum = sum(buf[16:]) & 0xFFFF
    _push_u16_be(buf, checksum)

    return bytes(buf)


def write_ot_file(
    path: Path,
    slices: list[OTSlice],
    total_samples: int,
    sample_rate: int = OT_SAMPLE_RATE,
    tempo: int = OT_DEFAULT_BPM,
) -> None:
    """Write an Octatrack .ot slice metadata file."""
    data = generate_ot_data(slices, total_samples, sample_rate, tempo)
    with open(str(path), "wb") as f:
        f.write(data)
    log.info("Wrote .ot file: %s (%d slices)", path, len(slices))


# ---------------------------------------------------------------------------
# Sample chain builder
# ---------------------------------------------------------------------------


def select_random(
    wav_paths: list[Path],
    count: int = OT_MAX_SLICES,
    seed: int | None = None,
) -> list[Path]:
    """Randomly select up to *count* files from a list.

    Preserves sorted order of the selected subset for reproducible chains.
    """
    if len(wav_paths) <= count:
        return list(wav_paths)
    rng = random.Random(seed)
    selected = set(rng.sample(range(len(wav_paths)), count))
    return [p for i, p in enumerate(wav_paths) if i in selected]


def build_chain(
    wav_paths: list[Path],
    output_wav: Path,
    evenly_spaced: bool = False,
    sample_rate: int = OT_SAMPLE_RATE,
    tempo: int = OT_DEFAULT_BPM,
    auto_convert_enabled: bool = False,
    normalize: bool = False,
    random_select: bool = False,
    random_seed: int | None = None,
) -> list[OTSlice]:
    """Concatenate WAV files into a sample chain and write the .ot file.

    Args:
        wav_paths: Ordered list of input WAV files.
        output_wav: Path for the concatenated output WAV.
        evenly_spaced: If True, zero-pad shorter samples to match the longest.
        sample_rate: Target sample rate for the chain.
        tempo: BPM to embed in the .ot file.
        auto_convert_enabled: If True, auto-convert non-conforming files
            (stereo, wrong sample rate/bit depth) to mono 16-bit at sample_rate.
            If False, non-conforming files raise ValueError.
        normalize: If True, peak-normalize each sample to -0.1 dBFS before
            concatenation for consistent volume across slices.
        random_select: If True, randomly select up to 64 files from wav_paths.
        random_seed: Seed for random selection (for reproducibility).

    Returns:
        List of OTSlice objects describing the chain.
    """
    if not wav_paths:
        raise ValueError("No input WAV files provided")

    if random_select:
        wav_paths = select_random(wav_paths, OT_MAX_SLICES, random_seed)
        log.info("Randomly selected %d files", len(wav_paths))

    if len(wav_paths) > OT_MAX_SLICES:
        raise ValueError(
            f"Too many files ({len(wav_paths)}); Octatrack supports max {OT_MAX_SLICES} slices"
        )

    # Read all files, auto-converting if needed
    file_samples: list[np.ndarray] = []
    max_len = 0
    tmp_files: list[Path] = []
    try:
        for p in wav_paths:
            read_path = p
            if auto_convert_enabled and needs_conversion(p, sample_rate):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                tmp_files.append(tmp_path)
                read_path = auto_convert(p, tmp_path, sample_rate)

            samples = read_wav_mono16(read_path, sample_rate)
            if normalize:
                samples = normalize_samples(samples)
            file_samples.append(samples)
            if len(samples) > max_len:
                max_len = len(samples)
            log.info("Read %s: %d samples", p.name, len(samples))
    finally:
        for tmp_path in tmp_files:
            tmp_path.unlink(missing_ok=True)

    # Build concatenated buffer and slice list
    slices: list[OTSlice] = []
    chain_parts: list[np.ndarray] = []
    offset = 0

    for samples in file_samples:
        sample_len = len(samples)
        if evenly_spaced:
            padded = np.zeros(max_len, dtype=np.int16)
            padded[:sample_len] = samples
            chain_parts.append(padded)
            slices.append(OTSlice(offset, max_len))
            offset += max_len
        else:
            chain_parts.append(samples)
            slices.append(OTSlice(offset, sample_len))
            offset += sample_len

    chain = np.concatenate(chain_parts)
    total_samples = len(chain)

    # Write output WAV
    write_wav_mono16(output_wav, chain, sample_rate)
    log.info(
        "Wrote chain WAV: %s (%d samples, %.2fs)",
        output_wav,
        total_samples,
        total_samples / sample_rate,
    )

    # Write .ot sidecar file
    ot_path = output_wav.with_suffix(".ot")
    write_ot_file(ot_path, slices, total_samples, sample_rate, tempo)

    return slices


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ot_utils",
        description="Create Octatrack-compatible sample chains with .ot slice files.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")

    sub = parser.add_subparsers(dest="command", required=True)

    ch = sub.add_parser("chain", help="build sample chain from a folder of WAVs")
    ch.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="input directory containing WAV files",
    )
    ch.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="output WAV file path (the .ot file is created alongside)",
    )
    ch.add_argument(
        "--even",
        action="store_true",
        help="zero-pad shorter samples to match the longest (evenly spaced slices)",
    )
    ch.add_argument(
        "--bpm",
        type=int,
        default=OT_DEFAULT_BPM,
        help=f"tempo in BPM for the .ot metadata (default: {OT_DEFAULT_BPM})",
    )
    ch.add_argument(
        "--sr",
        type=int,
        default=OT_SAMPLE_RATE,
        help=f"expected sample rate in Hz (default: {OT_SAMPLE_RATE})",
    )
    ch.add_argument(
        "--auto-convert",
        action="store_true",
        help="auto-convert non-conforming files (stereo, wrong rate/depth) to mono 16-bit",
    )
    ch.add_argument(
        "--normalize",
        action="store_true",
        help="peak-normalize each sample to -0.1 dBFS before chaining",
    )
    ch.add_argument(
        "--random",
        action="store_true",
        help="randomly select up to 64 samples from the input folder",
    )
    ch.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed for random selection (for reproducibility)",
    )
    ch.set_defaults(func=_cmd_chain)

    sl = sub.add_parser("slice", help="slice a single audio file into an OT chain")
    sl.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="input audio file to slice",
    )
    sl.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="output WAV file path (the .ot file is created alongside)",
    )
    slice_mode = sl.add_mutually_exclusive_group()
    slice_mode.add_argument(
        "--bpm",
        type=float,
        default=None,
        help="slice at beat boundaries for this BPM",
    )
    slice_mode.add_argument(
        "-n",
        "--count",
        type=int,
        default=None,
        help="divide audio into N equal slices",
    )
    slice_mode.add_argument(
        "--split",
        action="store_true",
        help="split at silence gaps instead of slicing",
    )
    sl.add_argument(
        "--beats",
        type=int,
        default=1,
        help="beats per slice when using --bpm (default: 1)",
    )
    sl.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.3,
        help="onset threshold 0.0-1.0 (default: 0.3, used when no --bpm, -n, or --split)",
    )
    sl.add_argument(
        "--sensitivity",
        type=float,
        default=1.5,
        help="onset strictness 1.0-3.0 (default: 1.5)",
    )
    sl.add_argument(
        "-m",
        "--method",
        choices=["hfc", "flux", "energy", "complex", "superflux"],
        default="hfc",
        help="onset algorithm (default: hfc)",
    )
    sl.add_argument(
        "--tempo",
        type=int,
        default=OT_DEFAULT_BPM,
        help=f"tempo for .ot metadata (default: {OT_DEFAULT_BPM})",
    )
    sl.add_argument(
        "--sr",
        type=int,
        default=OT_SAMPLE_RATE,
        help=f"sample rate in Hz (default: {OT_SAMPLE_RATE})",
    )
    sl.add_argument(
        "--auto-convert",
        action="store_true",
        help="auto-convert sliced files to mono 16-bit",
    )
    sl.add_argument(
        "--normalize",
        action="store_true",
        help="peak-normalize each slice to -0.1 dBFS",
    )
    sl.set_defaults(func=_cmd_slice)

    return parser


def _cmd_slice(args: argparse.Namespace) -> None:
    from wavtools.slicer import (
        slice_file_by_bpm,
        slice_file_by_count,
        slice_file_by_onsets,
        split_file_by_silence,
    )

    input_file: Path = args.input
    if not input_file.is_file():
        raise SystemExit(f"Error: {input_file} is not a file")

    # Slice into temporary directory, then chain the results
    tmp_dir = args.output.parent / f".{args.output.stem}_slices"
    try:
        if args.split:
            slice_paths = split_file_by_silence(input_file, tmp_dir)
        elif args.bpm is not None:
            slice_paths = slice_file_by_bpm(
                input_file,
                tmp_dir,
                args.bpm,
                args.beats,
            )
        elif args.count is not None:
            slice_paths = slice_file_by_count(input_file, tmp_dir, args.count)
        else:
            slice_paths = slice_file_by_onsets(
                input_file,
                tmp_dir,
                threshold=args.threshold,
                sensitivity=args.sensitivity,
                method=args.method,
            )

        if not slice_paths:
            raise SystemExit("Error: no slices produced")

        log.info("Chaining %d slices into %s", len(slice_paths), args.output)
        build_chain(
            sorted(slice_paths),
            args.output,
            sample_rate=args.sr,
            tempo=args.tempo,
            auto_convert_enabled=args.auto_convert,
            normalize=args.normalize,
        )
    finally:
        # Clean up temporary slice files
        if tmp_dir.exists():
            for f in tmp_dir.iterdir():
                f.unlink(missing_ok=True)
            tmp_dir.rmdir()


def _cmd_chain(args: argparse.Namespace) -> None:
    input_dir: Path = args.input
    if not input_dir.is_dir():
        raise SystemExit(f"Error: {input_dir} is not a directory")

    wav_files = sorted(input_dir.glob("*.wav"))
    if not wav_files:
        raise SystemExit(f"Error: no .wav files found in {input_dir}")

    log.info("Found %d WAV files in %s", len(wav_files), input_dir)
    build_chain(
        wav_files,
        args.output,
        evenly_spaced=args.even,
        sample_rate=args.sr,
        tempo=args.bpm,
        auto_convert_enabled=args.auto_convert,
        normalize=args.normalize,
        random_select=args.random,
        random_seed=args.seed,
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
