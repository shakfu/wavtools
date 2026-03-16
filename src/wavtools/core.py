"""Shared WAV I/O utilities for mono 16-bit PCM audio.

Functions in this module are used by multiple device-specific modules
(Octatrack, 2hp Play) that share the same audio format requirements:
mono, 16-bit, typically 44.1 kHz.
"""

from __future__ import annotations

import logging
import struct
from pathlib import Path

import cysox
import numpy as np

log = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 44100
DEFAULT_NORMALIZE_DB = -0.1


def needs_conversion(path: Path, sample_rate: int = DEFAULT_SAMPLE_RATE) -> bool:
    """Check whether a WAV file needs conversion to mono 16-bit."""
    meta = cysox.info(str(path))
    return bool(meta.channels != 1 or meta.sample_rate != sample_rate or meta.bits_per_sample != 16)


def auto_convert(
    path: Path,
    output_path: Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> Path:
    """Convert an audio file to mono 16-bit at the target sample rate.

    If the file already conforms, it is returned as-is (*output_path* is not
    written).  Otherwise cysox is used and *output_path* is returned.
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


def normalize_samples(samples: np.ndarray, target_db: float = DEFAULT_NORMALIZE_DB) -> np.ndarray:
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


def read_wav_mono16(path: Path, sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
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
    samples = (raw >> 16).astype(np.int16)
    return samples


def write_wav_mono16(
    path: Path,
    samples: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> None:
    """Write a 16-bit mono PCM WAV file with a minimal RIFF header."""
    samples = np.ascontiguousarray(samples, dtype=np.int16)
    n_samples = len(samples)
    data_bytes = n_samples * 2

    with open(str(path), "wb") as f:
        f.write(struct.pack("<4sI4s", b"RIFF", 0, b"WAVE"))
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
        f.write(struct.pack("<4sI", b"data", data_bytes))
        f.write(samples.tobytes())
        file_size = f.tell()
        f.seek(4)
        f.write(struct.pack("<I", file_size - 8))
