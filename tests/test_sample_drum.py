import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from wavtools.sample_drum import (
    SD_SAMPLE_RATE,
    auto_convert,
    build_parser,
    collect_wav_files,
    estimate_duration,
    estimate_ram_usage,
    extract_channel,
    needs_conversion,
    prepare_folder,
    ram_bytes,
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


def _make_stereo16_wav(path: Path, sr: int, n_frames: int) -> None:
    """Create a stereo 16-bit WAV with distinct L/R channels.

    Left channel is a 440 Hz sine, right channel is a 880 Hz sine.
    """
    left = (np.sin(np.linspace(0, 2 * np.pi * 440, n_frames)) * 16000).astype(np.int16)
    right = (np.sin(np.linspace(0, 2 * np.pi * 880, n_frames)) * 8000).astype(np.int16)
    stereo = np.column_stack([left, right]).flatten()
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
# extract_channel
# ---------------------------------------------------------------------------


class TestExtractChannel:
    def test_extract_left(self, tmp_path: Path):
        src = tmp_path / "stereo.wav"
        out = tmp_path / "mono.wav"
        _make_stereo16_wav(src, SD_SAMPLE_RATE, 4800)
        extract_channel(src, out, channel="L")

        with wave.open(str(out), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == SD_SAMPLE_RATE
            assert wf.getnframes() == 4800

    def test_extract_right(self, tmp_path: Path):
        src = tmp_path / "stereo.wav"
        out_l = tmp_path / "left.wav"
        out_r = tmp_path / "right.wav"
        _make_stereo16_wav(src, SD_SAMPLE_RATE, 4800)
        extract_channel(src, out_l, channel="L")
        extract_channel(src, out_r, channel="R")

        # L and R should differ (different frequencies)
        from wavtools.core import read_wav_mono16

        left = read_wav_mono16(out_l, SD_SAMPLE_RATE)
        right = read_wav_mono16(out_r, SD_SAMPLE_RATE)
        assert not np.array_equal(left, right)

    def test_extract_with_resample(self, tmp_path: Path):
        src = tmp_path / "stereo_44k.wav"
        out = tmp_path / "mono.wav"
        _make_stereo16_wav(src, 44100, 4410)
        extract_channel(src, out, channel="L", sample_rate=SD_SAMPLE_RATE)

        with wave.open(str(out), "rb") as wf:
            assert wf.getframerate() == SD_SAMPLE_RATE

    def test_invalid_channel_raises(self, tmp_path: Path):
        src = tmp_path / "stereo.wav"
        out = tmp_path / "mono.wav"
        _make_stereo16_wav(src, SD_SAMPLE_RATE, 100)
        with pytest.raises(ValueError, match="channel"):
            extract_channel(src, out, channel="M")

    def test_mono_input_raises(self, tmp_path: Path):
        src = tmp_path / "mono.wav"
        out = tmp_path / "out.wav"
        _make_mono16_wav(src, SD_SAMPLE_RATE, 100)
        with pytest.raises(ValueError, match="stereo"):
            extract_channel(src, out, channel="L")


# ---------------------------------------------------------------------------
# needs_conversion
# ---------------------------------------------------------------------------


class TestNeedsConversion:
    def test_conforming(self, tmp_path: Path):
        p = tmp_path / "ok.wav"
        _make_mono16_wav(p, SD_SAMPLE_RATE, 1000)
        assert needs_conversion(p) is False

    def test_stereo(self, tmp_path: Path):
        p = tmp_path / "stereo.wav"
        _make_stereo16_wav(p, SD_SAMPLE_RATE, 1000)
        assert needs_conversion(p) is True

    def test_wrong_rate(self, tmp_path: Path):
        p = tmp_path / "44k.wav"
        _make_mono16_wav(p, 44100, 1000)
        assert needs_conversion(p) is True

    def test_wrong_bits(self, tmp_path: Path):
        p = tmp_path / "24bit.wav"
        _make_mono_24bit_wav(p, SD_SAMPLE_RATE, 1000)
        assert needs_conversion(p) is True


# ---------------------------------------------------------------------------
# auto_convert
# ---------------------------------------------------------------------------


class TestAutoConvert:
    def test_returns_original_when_conforming(self, tmp_path: Path):
        p = tmp_path / "ok.wav"
        out = tmp_path / "out.wav"
        _make_mono16_wav(p, SD_SAMPLE_RATE, 1000)
        result = auto_convert(p, out)
        assert result == p
        assert not out.exists()

    def test_converts_stereo_extracts_left(self, tmp_path: Path):
        p = tmp_path / "stereo.wav"
        out = tmp_path / "out.wav"
        _make_stereo16_wav(p, SD_SAMPLE_RATE, 1000)
        result = auto_convert(p, out, stereo_channel="L")
        assert result == out
        import cysox

        meta = cysox.info(str(out))
        assert meta.channels == 1
        assert meta.bits_per_sample == 16

    def test_converts_stereo_extracts_right(self, tmp_path: Path):
        p = tmp_path / "stereo.wav"
        out_l = tmp_path / "out_l.wav"
        out_r = tmp_path / "out_r.wav"
        _make_stereo16_wav(p, SD_SAMPLE_RATE, 1000)
        auto_convert(p, out_l, stereo_channel="L")
        auto_convert(p, out_r, stereo_channel="R")
        from wavtools.core import read_wav_mono16

        left = read_wav_mono16(out_l, SD_SAMPLE_RATE)
        right = read_wav_mono16(out_r, SD_SAMPLE_RATE)
        assert not np.array_equal(left, right)

    def test_converts_wrong_rate(self, tmp_path: Path):
        p = tmp_path / "44k.wav"
        out = tmp_path / "out.wav"
        _make_mono16_wav(p, 44100, 1000)
        result = auto_convert(p, out)
        assert result == out
        import cysox

        meta = cysox.info(str(out))
        assert meta.sample_rate == SD_SAMPLE_RATE


# ---------------------------------------------------------------------------
# RAM estimation
# ---------------------------------------------------------------------------


class TestRamBytes:
    def test_basic(self):
        assert ram_bytes(48000) == 96000  # 1 second at 48kHz 16-bit

    def test_zero(self):
        assert ram_bytes(0) == 0


class TestEstimateRamUsage:
    def test_single_file(self, tmp_path: Path):
        p = tmp_path / "test.wav"
        _make_mono16_wav(p, SD_SAMPLE_RATE, 48000)  # 1 second
        usage = estimate_ram_usage([p])
        assert usage == 96000

    def test_multiple_files(self, tmp_path: Path):
        paths = []
        for i in range(3):
            p = tmp_path / f"s{i}.wav"
            _make_mono16_wav(p, SD_SAMPLE_RATE, 48000)
            paths.append(p)
        usage = estimate_ram_usage(paths)
        assert usage == 96000 * 3


class TestEstimateDuration:
    def test_one_second(self, tmp_path: Path):
        p = tmp_path / "test.wav"
        _make_mono16_wav(p, SD_SAMPLE_RATE, 48000)
        duration = estimate_duration([p])
        assert abs(duration - 1.0) < 0.01

    def test_multiple_files(self, tmp_path: Path):
        paths = []
        for i in range(5):
            p = tmp_path / f"s{i}.wav"
            _make_mono16_wav(p, SD_SAMPLE_RATE, 48000)
            paths.append(p)
        duration = estimate_duration(paths)
        assert abs(duration - 5.0) < 0.01


# ---------------------------------------------------------------------------
# collect_wav_files
# ---------------------------------------------------------------------------


class TestCollectWavFiles:
    def test_collects_alphabetically(self, tmp_path: Path):
        for name in ["c.wav", "a.wav", "b.wav"]:
            _make_mono16_wav(tmp_path / name, SD_SAMPLE_RATE, 100)
        result = collect_wav_files(tmp_path)
        assert [p.name for p in result] == ["a.wav", "b.wav", "c.wav"]

    def test_excludes_non_wav(self, tmp_path: Path):
        _make_mono16_wav(tmp_path / "ok.wav", SD_SAMPLE_RATE, 100)
        (tmp_path / "readme.txt").write_text("hi")
        result = collect_wav_files(tmp_path)
        assert len(result) == 1

    def test_empty_dir(self, tmp_path: Path):
        assert collect_wav_files(tmp_path) == []


# ---------------------------------------------------------------------------
# prepare_folder
# ---------------------------------------------------------------------------


class TestPrepareFolder:
    def test_basic(self, tmp_path: Path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        for name in ["kick.wav", "snare.wav"]:
            _make_mono16_wav(src / name, SD_SAMPLE_RATE, 4800)
        wav_files = collect_wav_files(src)
        result = prepare_folder(wav_files, dst)
        assert len(result) == 2
        assert all(p.exists() for p in result)

    def test_preserves_filenames(self, tmp_path: Path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        _make_mono16_wav(src / "kick_01.wav", SD_SAMPLE_RATE, 100)
        wav_files = collect_wav_files(src)
        result = prepare_folder(wav_files, dst)
        assert result[0].name == "kick_01.wav"

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="No input"):
            prepare_folder([], Path("/nonexistent"))

    def test_normalize(self, tmp_path: Path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        # Write a quiet WAV
        quiet = np.array([100, -100, 50, -50] * 100, dtype=np.int16)
        p = src / "quiet.wav"
        with wave.open(str(p), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SD_SAMPLE_RATE)
            wf.writeframes(quiet.tobytes())
        wav_files = collect_wav_files(src)
        prepare_folder(wav_files, dst, normalize=True)
        from wavtools.core import read_wav_mono16

        result = read_wav_mono16(dst / "quiet.wav", SD_SAMPLE_RATE)
        assert np.max(np.abs(result)) > 100

    def test_ram_validation_passes(self, tmp_path: Path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        # Small file, well under 32MB
        _make_mono16_wav(src / "small.wav", SD_SAMPLE_RATE, 4800)
        wav_files = collect_wav_files(src)
        # Should not raise
        prepare_folder(wav_files, dst, validate_ram=True)

    def test_ram_validation_fails(self, tmp_path: Path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        # Create files that exceed 32MB
        # 32MB / 2 bytes per sample = 16,777,216 samples
        # Make a few large files that sum to > 32MB
        for i in range(4):
            p = src / f"big_{i}.wav"
            _make_mono16_wav(p, SD_SAMPLE_RATE, 5_000_000)  # ~10MB each
        wav_files = collect_wav_files(src)
        with pytest.raises(ValueError, match="RAM limit"):
            prepare_folder(wav_files, dst, validate_ram=True)

    def test_skip_ram_validation(self, tmp_path: Path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        for i in range(4):
            p = src / f"big_{i}.wav"
            _make_mono16_wav(p, SD_SAMPLE_RATE, 5_000_000)
        wav_files = collect_wav_files(src)
        # Should not raise with validation disabled
        result = prepare_folder(wav_files, dst, validate_ram=False)
        assert len(result) == 4


class TestPrepareFolderAutoConvert:
    def test_converts_stereo(self, tmp_path: Path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        _make_stereo16_wav(src / "stereo.wav", SD_SAMPLE_RATE, 1000)
        wav_files = collect_wav_files(src)
        result = prepare_folder(wav_files, dst, auto_convert_enabled=True)
        assert len(result) == 1
        import cysox

        meta = cysox.info(str(result[0]))
        assert meta.channels == 1
        assert meta.bits_per_sample == 16

    def test_converts_wrong_rate(self, tmp_path: Path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        _make_mono16_wav(src / "44k.wav", 44100, 1000)
        wav_files = collect_wav_files(src)
        result = prepare_folder(wav_files, dst, auto_convert_enabled=True)
        assert len(result) == 1
        import cysox

        meta = cysox.info(str(result[0]))
        assert meta.sample_rate == SD_SAMPLE_RATE

    def test_stereo_right_channel(self, tmp_path: Path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        _make_stereo16_wav(src / "stereo.wav", SD_SAMPLE_RATE, 1000)
        wav_files = collect_wav_files(src)
        prepare_folder(wav_files, dst, auto_convert_enabled=True, stereo_channel="R")
        import cysox

        meta = cysox.info(str(dst / "stereo.wav"))
        assert meta.channels == 1

    def test_lower_sample_rate_for_ram_saving(self, tmp_path: Path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        _make_mono16_wav(src / "test.wav", 22050, 2205)
        wav_files = collect_wav_files(src)
        # Convert to 22050 (not 48000) to save RAM
        result = prepare_folder(wav_files, dst, sample_rate=22050, auto_convert_enabled=True)
        assert len(result) == 1
        import cysox

        meta = cysox.info(str(result[0]))
        assert meta.sample_rate == 22050


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------


class TestBuildParser:
    def test_prepare_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["prepare", "-i", "input/", "-o", "SAMPLES/kit"])
        assert args.command == "prepare"
        assert args.input == Path("input/")
        assert args.output == Path("SAMPLES/kit")

    def test_prepare_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["prepare", "-i", "in/", "-o", "out/"])
        assert args.sr == SD_SAMPLE_RATE
        assert args.auto_convert is False
        assert args.channel == "L"
        assert args.normalize is False
        assert args.no_validate is False
        assert args.verbose is False

    def test_prepare_all_flags(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "-v",
                "prepare",
                "-i",
                "in/",
                "-o",
                "out/",
                "--sr",
                "22050",
                "--auto-convert",
                "--channel",
                "R",
                "--normalize",
                "--no-validate",
            ]
        )
        assert args.sr == 22050
        assert args.auto_convert is True
        assert args.channel == "R"
        assert args.normalize is True
        assert args.no_validate is True
        assert args.verbose is True
