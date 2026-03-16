#!/usr/bin/env python3
"""Morphagene reel utility.

Create Morphagene-compatible 32-bit float / 48 kHz WAV reels with splice
markers derived from Ableton Live project files or automatic onset detection.

Usage:
    mg_utils ableton -w input.wav -l project.als -o output.wav
    mg_utils onset -w input.wav -o output.wav [-s 50]
"""

from __future__ import annotations

import argparse
import gzip
import logging
import struct
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import cysox
import numpy as np
from cysox import onset

log = logging.getLogger(__name__)

MORPHAGENE_SAMPLE_RATE = 48000
MORPHAGENE_MAX_SPLICES = 300
MORPHAGENE_MAX_DURATION_MIN = 2.9


# ---------------------------------------------------------------------------
# WAV I/O
# ---------------------------------------------------------------------------


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a WAV file, returning float32 (channels, samples) and sample rate.

    IEEE-float WAVs are read via the stdlib ``wave`` module to avoid the
    precision loss that would occur from cysox's int32 sample pipeline.
    All other formats are read via cysox (which supports anything libsox
    handles: mp3, flac, ogg, aiff, etc.).
    """
    meta = cysox.info(str(path))

    if meta.encoding == "float":
        return _read_wav_float(path, meta)

    chunks = []
    for chunk in cysox.stream(str(path)):
        chunks.append(np.frombuffer(chunk, dtype=np.int32).copy())
    raw = np.concatenate(chunks)
    audio = raw.astype(np.float32) / 2147483648.0
    audio = audio.reshape(-1, meta.channels).T
    return audio, meta.sample_rate


def _read_wav_float(path: Path, meta: cysox.AudioInfo) -> tuple[np.ndarray, int]:
    """Read float32 samples directly from the WAV data chunk.

    Both cysox (int32 pipeline) and the stdlib wave module can lose
    precision for IEEE-float WAVs, so we parse the RIFF structure ourselves
    and pull the raw float32 bytes out of the ``data`` chunk.
    """
    with open(str(path), "rb") as f:
        f.read(12)  # skip RIFF header + WAVE id
        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                raise ValueError("data chunk not found in WAV file")
            chunk_size = struct.unpack("<I", f.read(4))[0]
            if chunk_id == b"data":
                raw = f.read(chunk_size)
                break
            f.seek(chunk_size + (chunk_size % 2), 1)  # skip chunk + pad byte
    audio = np.frombuffer(raw, dtype=np.float32).copy()
    audio = audio.reshape(-1, meta.channels).T
    return audio, meta.sample_rate


def write_wav(
    path: Path,
    audio: np.ndarray,
    sample_rate: int,
    markers: list[dict[str, int | str]] | None = None,
) -> None:
    """Write a 32-bit float WAV with optional cue markers.

    cysox does not expose cue-point writing, so this uses direct struct
    packing to produce a WAV with embedded ``cue`` and ``LIST/adtl/labl``
    chunks that the Morphagene reads as splice positions.

    Args:
        path: Output file path.
        audio: Float32 array shaped (channels, samples).
        sample_rate: Sample rate in Hz.
        markers: List of ``{'position': <sample_offset>, 'label': <str>}`` dicts.
    """
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    channels, samples = audio.shape
    data_bytes = channels * samples * 4

    with open(str(path), "wb") as f:
        # RIFF header -- size placeholder, patched at end
        f.write(struct.pack("<4sI4s", b"RIFF", 0, b"WAVE"))

        # fmt chunk (IEEE float)
        f.write(
            struct.pack(
                "<4sIHHIIHH",
                b"fmt ",
                16,
                3,
                channels,
                sample_rate,
                sample_rate * channels * 4,
                channels * 4,
                32,
            )
        )

        # data chunk
        f.write(struct.pack("<4sI", b"data", data_bytes))
        interleaved = np.ascontiguousarray(audio.T.flatten())
        f.write(interleaved.tobytes())

        if markers:
            positions = [m["position"] for m in markers]
            labels = [m.get("label", "") for m in markers]

            # cue chunk
            f.write(b"cue ")
            f.write(struct.pack("<ii", 4 + 24 * len(positions), len(positions)))
            for i, pos in enumerate(positions):
                f.write(struct.pack("<iiiiii", i + 1, pos, 0x64617461, 0, 0, pos))

            # LIST / adtl with labl sub-chunks
            labl_data = b""
            for i, lbl in enumerate(labels):
                encoded = str(lbl).encode("ascii") + b"\x00"
                if len(encoded) % 2 == 1:
                    encoded += b"\x00"
                labl_data += b"labl" + struct.pack("<ii", len(encoded) + 4, i + 1) + encoded

            f.write(b"LIST")
            f.write(struct.pack("<i", len(labl_data) + 4))
            f.write(b"adtl")
            f.write(labl_data)

        # Patch RIFF size now that total file size is known
        file_size = f.tell()
        f.seek(4)
        f.write(struct.pack("<I", file_size - 8))


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


def resample(audio: np.ndarray, old_rate: int, new_rate: int) -> np.ndarray:
    """Resample audio (channels, samples) via per-channel linear interpolation.

    This is a lightweight fallback for in-memory arrays.  The main pipeline
    uses :func:`resample_file` instead, which delegates to libsox's polyphase
    resampler for higher quality.
    """
    if old_rate == new_rate:
        return audio
    old_len = audio.shape[1]
    new_len = int(old_len * new_rate / old_rate)
    old_t = np.linspace(0, 1, old_len)
    new_t = np.linspace(0, 1, new_len)
    return np.array(
        [np.interp(new_t, old_t, ch) for ch in audio],
        dtype=np.float32,
    )


def resample_file(src: Path, dst: Path, target_rate: int) -> None:
    """Resample an audio file to *target_rate* using libsox's polyphase resampler."""
    cysox.convert(str(src), str(dst), sample_rate=target_rate)


# ---------------------------------------------------------------------------
# Marker sources
# ---------------------------------------------------------------------------


def load_ableton_markers(als_path: Path) -> np.ndarray:
    """Extract locator positions (seconds) from a gzipped Ableton .als file."""
    with gzip.open(str(als_path), "rb") as f:
        root = ET.fromstring(f.read())

    bpm: float | None = None
    for tempo in root.iter("Tempo"):
        manual = tempo.find("Manual")
        if manual is not None:
            bpm = float(manual.get("Value", "0"))
            break

    if bpm is None:
        raise ValueError("No BPM found in Ableton project file")

    bps = bpm / 60.0
    log.info("BPM: %.1f  BPS: %.1f", bpm, bps)

    markers_sec: list[float] = []
    for locator in root.iter("Locator"):
        time_el = locator.find("Time")
        if time_el is not None:
            beat_time = float(time_el.get("Value", "nan"))
            sec = beat_time / bps
            markers_sec.append(sec)
            log.info("Locator %s at %.3fs", locator.get("Id"), sec)

    return np.array(markers_sec, dtype=np.float64)


def detect_onsets(
    wav_path: Path,
    threshold: float = 0.3,
    sensitivity: float = 1.5,
    method: str = "hfc",
) -> np.ndarray:
    """Detect onsets using cysox onset detection.

    Args:
        wav_path: Path to input WAV file.
        threshold: Detection threshold 0.0-1.0 (lower = more sensitive).
        sensitivity: Detection strictness 1.0-3.0 (higher = stricter).
        method: Algorithm -- "hfc", "flux", "energy", "complex", or "superflux".

    Returns:
        Unique onset times in seconds.
    """
    onsets = onset.detect(
        str(wav_path),
        threshold=threshold,
        sensitivity=sensitivity,
        method=method,
    )
    return np.unique(np.array(onsets, dtype=np.float64))


def select_markers(onsets: np.ndarray, count: int) -> np.ndarray:
    """Evenly select *count* markers from a larger onset array.

    First marker is forced to 0.0 so the reel starts at the beginning.
    """
    if count >= len(onsets):
        return onsets
    k, m = divmod(len(onsets), count)
    selected = [onsets[i * k + min(i, m)] for i in range(count)]
    selected[0] = 0.0
    return np.array(selected)


# ---------------------------------------------------------------------------
# Shared pipeline
# ---------------------------------------------------------------------------


def make_reel(
    wav_path: Path,
    output_path: Path,
    markers_sec: np.ndarray,
) -> None:
    """Read *wav_path*, resample to 48 kHz, attach *markers_sec*, write reel."""
    meta = cysox.info(str(wav_path))
    sample_rate = meta.sample_rate

    if sample_rate != MORPHAGENE_SAMPLE_RATE:
        log.info("Resampling %d Hz -> %d Hz (libsox)", sample_rate, MORPHAGENE_SAMPLE_RATE)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            resample_file(wav_path, tmp_path, MORPHAGENE_SAMPLE_RATE)
            audio, _ = read_wav(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
    else:
        audio, _ = read_wav(wav_path)

    log.info(
        "Input: %d ch, %d samples, %d Hz",
        audio.shape[0],
        audio.shape[1],
        MORPHAGENE_SAMPLE_RATE,
    )

    frame_positions = (markers_sec * MORPHAGENE_SAMPLE_RATE).astype(np.int32)
    marker_dicts: list[dict[str, int | str]] = [
        {"position": int(pos), "label": f"marker{i + 1}"} for i, pos in enumerate(frame_positions)
    ]

    duration_min = audio.shape[1] / MORPHAGENE_SAMPLE_RATE / 60.0
    if len(marker_dicts) > MORPHAGENE_MAX_SPLICES:
        log.warning(
            "Splice count (%d) exceeds Morphagene limit (%d)",
            len(marker_dicts),
            MORPHAGENE_MAX_SPLICES,
        )
    if duration_min > MORPHAGENE_MAX_DURATION_MIN:
        log.warning(
            "Duration (%.1f min) exceeds Morphagene limit (%.1f min)",
            duration_min,
            MORPHAGENE_MAX_DURATION_MIN,
        )

    write_wav(output_path, audio, MORPHAGENE_SAMPLE_RATE, markers=marker_dicts)
    log.info("Wrote %d markers to %s", len(marker_dicts), output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_ableton(args: argparse.Namespace) -> None:
    markers_sec = load_ableton_markers(args.labels)
    make_reel(args.wav, args.output, markers_sec)


def _cmd_onset(args: argparse.Namespace) -> None:
    onsets = detect_onsets(
        args.wav,
        threshold=args.threshold,
        sensitivity=args.sensitivity,
        method=args.method,
    )
    if args.splices is not None:
        onsets = select_markers(onsets, args.splices)
    make_reel(args.wav, args.output, onsets)


def _cmd_slice(args: argparse.Namespace) -> None:
    from .slicer import (
        slice_points_by_bpm,
        slice_points_by_count,
        slice_points_by_onsets,
    )

    if args.bpm is not None:
        markers = slice_points_by_bpm(args.wav, args.bpm, args.beats)
    elif args.count is not None:
        markers = slice_points_by_count(args.wav, args.count)
    else:
        markers = slice_points_by_onsets(
            args.wav,
            threshold=args.threshold,
            sensitivity=args.sensitivity,
            method=args.method,
        )
    if args.splices is not None:
        markers = select_markers(markers, args.splices)
    make_reel(args.wav, args.output, markers)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mg_utils",
        description="Create Morphagene-compatible WAV reels with splice markers.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    sub = parser.add_subparsers(dest="command", required=True)

    ab = sub.add_parser("ableton", help="markers from Ableton Live locators (.als)")
    ab.add_argument("-w", "--wav", type=Path, required=True, help="input WAV file")
    ab.add_argument("-l", "--labels", type=Path, required=True, help="Ableton .als project")
    ab.add_argument("-o", "--output", type=Path, required=True, help="output WAV reel")
    ab.set_defaults(func=_cmd_ableton)

    on = sub.add_parser("onset", help="markers from onset detection (cysox)")
    on.add_argument("-w", "--wav", type=Path, required=True, help="input WAV file")
    on.add_argument("-o", "--output", type=Path, required=True, help="output WAV reel")
    on.add_argument(
        "-s",
        "--splices",
        type=int,
        default=None,
        help="max splice markers (default: all detected onsets)",
    )
    on.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.3,
        help="detection threshold 0.0-1.0, lower = more sensitive (default: 0.3)",
    )
    on.add_argument(
        "--sensitivity",
        type=float,
        default=1.5,
        help="detection strictness 1.0-3.0, higher = stricter (default: 1.5)",
    )
    on.add_argument(
        "-m",
        "--method",
        choices=["hfc", "flux", "energy", "complex", "superflux"],
        default="hfc",
        help="onset algorithm (default: hfc)",
    )
    on.set_defaults(func=_cmd_onset)

    sl = sub.add_parser("slice", help="markers from BPM, count, or onset slicing (cysox)")
    sl.add_argument("-w", "--wav", type=Path, required=True, help="input WAV file")
    sl.add_argument("-o", "--output", type=Path, required=True, help="output WAV reel")
    sl.add_argument(
        "-s",
        "--splices",
        type=int,
        default=None,
        help="max splice markers (default: all detected)",
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
        help="onset threshold 0.0-1.0 (default: 0.3, used when no --bpm or -n)",
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
    sl.set_defaults(func=_cmd_slice)

    return parser


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
