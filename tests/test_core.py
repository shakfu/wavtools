import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from wavtools.core import (
    DEFAULT_SAMPLE_RATE,
    auto_convert,
    needs_conversion,
    normalize_samples,
    read_wav_mono16,
    write_wav_mono16,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mono16_wav(path: Path, sr: int, n_frames: int) -> None:
    """Create a mono 16-bit PCM WAV with a sine tone."""
    tone = (np.sin(np.linspace(0, 2 * np.pi * 440, n_frames)) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(tone.tobytes())


def _make_stereo_wav(path: Path, sr: int, n_frames: int) -> None:
    """Create a stereo 16-bit PCM WAV."""
    tone = (np.sin(np.linspace(0, 2 * np.pi * 440, n_frames)) * 32767).astype(np.int16)
    stereo = np.column_stack([tone, tone]).flatten()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(stereo.tobytes())


def _make_mono_24bit_wav(path: Path, sr: int, n_frames: int) -> None:
    """Create a mono 24-bit PCM WAV."""
    tone = (np.sin(np.linspace(0, 2 * np.pi * 440, n_frames)) * 8388607).astype(np.int32)
    raw = b""
    for s in tone:
        b = int(s) & 0xFFFFFF
        raw += struct.pack("<I", b)[:3]
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(3)
        wf.setframerate(sr)
        wf.writeframes(raw)


# ---------------------------------------------------------------------------
# needs_conversion
# ---------------------------------------------------------------------------


class TestNeedsConversion:
    def test_conforming(self, tmp_path: Path):
        p = tmp_path / "ok.wav"
        _make_mono16_wav(p, DEFAULT_SAMPLE_RATE, 1000)
        assert needs_conversion(p) is False

    def test_stereo(self, tmp_path: Path):
        p = tmp_path / "stereo.wav"
        _make_stereo_wav(p, DEFAULT_SAMPLE_RATE, 1000)
        assert needs_conversion(p) is True

    def test_wrong_rate(self, tmp_path: Path):
        p = tmp_path / "48k.wav"
        _make_mono16_wav(p, 48000, 1000)
        assert needs_conversion(p) is True

    def test_wrong_bits(self, tmp_path: Path):
        p = tmp_path / "24bit.wav"
        _make_mono_24bit_wav(p, DEFAULT_SAMPLE_RATE, 1000)
        assert needs_conversion(p) is True

    def test_custom_sample_rate(self, tmp_path: Path):
        p = tmp_path / "48k.wav"
        _make_mono16_wav(p, 48000, 1000)
        assert needs_conversion(p, sample_rate=48000) is False


# ---------------------------------------------------------------------------
# auto_convert
# ---------------------------------------------------------------------------


class TestAutoConvert:
    def test_returns_original_when_conforming(self, tmp_path: Path):
        p = tmp_path / "ok.wav"
        out = tmp_path / "out.wav"
        _make_mono16_wav(p, DEFAULT_SAMPLE_RATE, 1000)
        result = auto_convert(p, out)
        assert result == p
        assert not out.exists()

    def test_converts_stereo(self, tmp_path: Path):
        p = tmp_path / "stereo.wav"
        out = tmp_path / "out.wav"
        _make_stereo_wav(p, DEFAULT_SAMPLE_RATE, 1000)
        result = auto_convert(p, out)
        assert result == out
        assert out.exists()
        import cysox

        meta = cysox.info(str(out))
        assert meta.channels == 1
        assert meta.sample_rate == DEFAULT_SAMPLE_RATE
        assert meta.bits_per_sample == 16

    def test_converts_wrong_rate(self, tmp_path: Path):
        p = tmp_path / "48k.wav"
        out = tmp_path / "out.wav"
        _make_mono16_wav(p, 48000, 1000)
        result = auto_convert(p, out)
        assert result == out
        import cysox

        meta = cysox.info(str(out))
        assert meta.sample_rate == DEFAULT_SAMPLE_RATE


# ---------------------------------------------------------------------------
# read_wav_mono16
# ---------------------------------------------------------------------------


class TestReadWavMono16:
    def test_read_conforming(self, tmp_path: Path):
        p = tmp_path / "test.wav"
        _make_mono16_wav(p, DEFAULT_SAMPLE_RATE, 4410)
        samples = read_wav_mono16(p)
        assert samples.dtype == np.int16
        assert len(samples) == 4410

    def test_reject_stereo(self, tmp_path: Path):
        p = tmp_path / "stereo.wav"
        _make_stereo_wav(p, DEFAULT_SAMPLE_RATE, 1000)
        with pytest.raises(ValueError, match="mono"):
            read_wav_mono16(p)

    def test_reject_wrong_rate(self, tmp_path: Path):
        p = tmp_path / "48k.wav"
        _make_mono16_wav(p, 48000, 1000)
        with pytest.raises(ValueError, match="44100"):
            read_wav_mono16(p)


# ---------------------------------------------------------------------------
# write_wav_mono16
# ---------------------------------------------------------------------------


class TestWriteWavMono16:
    def test_roundtrip(self, tmp_path: Path):
        samples = (np.sin(np.linspace(0, 2 * np.pi * 440, 4410)) * 32767).astype(np.int16)
        p = tmp_path / "out.wav"
        write_wav_mono16(p, samples)

        with wave.open(str(p), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == DEFAULT_SAMPLE_RATE
            assert wf.getnframes() == 4410

    def test_riff_header(self, tmp_path: Path):
        samples = np.zeros(100, dtype=np.int16)
        p = tmp_path / "out.wav"
        write_wav_mono16(p, samples)

        with open(str(p), "rb") as f:
            riff = f.read(4)
            assert riff == b"RIFF"
            f.read(4)
            wave_tag = f.read(4)
            assert wave_tag == b"WAVE"

    def test_riff_size_patched(self, tmp_path: Path):
        samples = np.zeros(100, dtype=np.int16)
        p = tmp_path / "out.wav"
        write_wav_mono16(p, samples)

        with open(str(p), "rb") as f:
            f.read(4)  # RIFF
            (size,) = struct.unpack("<I", f.read(4))
            file_size = p.stat().st_size
            assert size == file_size - 8

    def test_custom_sample_rate(self, tmp_path: Path):
        samples = np.zeros(100, dtype=np.int16)
        p = tmp_path / "out.wav"
        write_wav_mono16(p, samples, sample_rate=48000)

        with wave.open(str(p), "rb") as wf:
            assert wf.getframerate() == 48000


# ---------------------------------------------------------------------------
# normalize_samples
# ---------------------------------------------------------------------------


class TestNormalizeSamples:
    def test_quiet_signal_boosted(self):
        samples = np.array([100, -100, 50, -50], dtype=np.int16)
        result = normalize_samples(samples)
        assert np.max(np.abs(result)) > np.max(np.abs(samples))

    def test_loud_signal(self):
        samples = np.array([32767, -32767, 16000], dtype=np.int16)
        result = normalize_samples(samples)
        assert result.dtype == np.int16

    def test_silent_signal(self):
        samples = np.zeros(100, dtype=np.int16)
        result = normalize_samples(samples)
        assert np.all(result == 0)

    def test_preserves_dtype(self):
        samples = np.array([1000, -1000], dtype=np.int16)
        result = normalize_samples(samples)
        assert result.dtype == np.int16

    def test_custom_target_db(self):
        samples = np.array([100, -100], dtype=np.int16)
        result_default = normalize_samples(samples)
        result_quiet = normalize_samples(samples, target_db=-6.0)
        assert np.max(np.abs(result_default)) > np.max(np.abs(result_quiet))
