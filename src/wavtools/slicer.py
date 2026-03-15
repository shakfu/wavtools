#!/usr/bin/env python3
"""General audio slicing services using cysox.

Provides slice point detection and file slicing for use with both
Morphagene (splice markers) and Octatrack (sample chains) workflows.

Slice points are returned as sorted arrays of times in seconds, suitable
for passing directly to :func:`wavtools.morphagene.make_reel`.

Sliced files are written to a directory, suitable for passing to
:func:`wavtools.octatrack.build_chain`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cysox
import numpy as np
from cysox import onset

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Slice point detection (returns times in seconds)
# ---------------------------------------------------------------------------


def slice_points_by_onsets(
    path: Path,
    threshold: float = 0.3,
    sensitivity: float = 1.5,
    method: str = "hfc",
) -> np.ndarray:
    """Detect slice points at audio transients.

    Args:
        path: Input audio file.
        threshold: Detection threshold 0.0-1.0 (lower = more sensitive).
        sensitivity: Peak-picking strictness 1.0-3.0 (higher = stricter).
        method: Onset algorithm -- "hfc", "flux", "energy", "complex",
            or "superflux".

    Returns:
        Sorted unique onset times in seconds.
    """
    onsets = onset.detect(
        str(path),
        threshold=threshold,
        sensitivity=sensitivity,
        method=method,
    )
    result = np.unique(np.array(onsets, dtype=np.float64))
    log.info("Detected %d onsets in %s", len(result), path.name)
    return result


def slice_points_by_bpm(
    path: Path,
    bpm: float,
    beats_per_slice: int = 1,
) -> np.ndarray:
    """Calculate slice points at beat boundaries.

    Args:
        path: Input audio file (used to determine duration).
        bpm: Tempo in beats per minute.
        beats_per_slice: Number of beats per slice (default: 1).

    Returns:
        Sorted slice times in seconds, starting at 0.0.
    """
    meta = cysox.info(str(path))
    duration = meta.duration
    beat_duration = 60.0 / bpm * beats_per_slice
    points = np.arange(0.0, duration, beat_duration)
    log.info(
        "BPM %.1f, %d beats/slice -> %d slice points over %.2fs",
        bpm,
        beats_per_slice,
        len(points),
        duration,
    )
    return points


def slice_points_by_count(
    path: Path,
    count: int,
) -> np.ndarray:
    """Divide audio into *count* equal slices.

    Args:
        path: Input audio file.
        count: Number of slices.

    Returns:
        Sorted slice times in seconds, starting at 0.0.
    """
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    meta = cysox.info(str(path))
    duration = meta.duration
    points = np.linspace(0.0, duration, count, endpoint=False)
    log.info("Divided %.2fs into %d equal slices", duration, count)
    return points


# ---------------------------------------------------------------------------
# File slicing (writes individual WAVs via cysox)
# ---------------------------------------------------------------------------


def slice_file_by_onsets(
    path: Path,
    output_dir: Path,
    threshold: float = 0.3,
    sensitivity: float = 1.5,
    method: str = "hfc",
) -> list[Path]:
    """Slice audio at detected transients, writing individual files.

    Uses :func:`cysox.slice_loop` with transient detection.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    result: list[str] = cysox.slice_loop(
        str(path),
        str(output_dir),
        threshold=threshold,
        sensitivity=sensitivity,
        onset_method=method,
    )
    paths = [Path(p) for p in result]
    log.info("Sliced %s into %d files (onsets)", path.name, len(paths))
    return paths


def slice_file_by_bpm(
    path: Path,
    output_dir: Path,
    bpm: float,
    beats_per_slice: int = 1,
) -> list[Path]:
    """Slice audio at beat boundaries, writing individual files.

    Uses :func:`cysox.slice_loop` with BPM-based slicing.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    result: list[str] = cysox.slice_loop(
        str(path),
        str(output_dir),
        bpm=bpm,
        beats_per_slice=beats_per_slice,
    )
    paths = [Path(p) for p in result]
    log.info("Sliced %s into %d files (BPM %.1f)", path.name, len(paths), bpm)
    return paths


def slice_file_by_count(
    path: Path,
    output_dir: Path,
    count: int,
) -> list[Path]:
    """Slice audio into *count* equal parts, writing individual files.

    Uses :func:`cysox.slice_loop` with count-based slicing.
    """
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    output_dir.mkdir(parents=True, exist_ok=True)
    result: list[str] = cysox.slice_loop(
        str(path),
        str(output_dir),
        slices=count,
    )
    paths = [Path(p) for p in result]
    log.info("Sliced %s into %d equal files", path.name, len(paths))
    return paths


def split_file_by_silence(
    path: Path,
    output_dir: Path,
    threshold_db: float = -48.0,
    min_silence: float = 0.25,
    min_segment: float = 0.25,
) -> list[Path]:
    """Split audio at silence gaps, writing individual files.

    Uses :func:`cysox.split_by_silence`.

    Args:
        path: Input audio file.
        output_dir: Directory for output segment files.
        threshold_db: Amplitude threshold in dB (default: -48).
        min_silence: Minimum silence duration in seconds to trigger a split.
        min_segment: Minimum segment duration in seconds (shorter segments
            are discarded).

    Returns:
        List of paths to segment files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    result: list[str] = cysox.split_by_silence(
        str(path),
        str(output_dir),
        threshold_db=threshold_db,
        min_silence=min_silence,
        min_segment=min_segment,
    )
    paths = [Path(p) for p in result]
    log.info("Split %s into %d segments (silence)", path.name, len(paths))
    return paths
