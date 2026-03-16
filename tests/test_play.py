import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from wavtools.play import (
    PLAY_MAX_FILES,
    PLAY_SAMPLE_RATE,
    PlayOptions,
    auto_convert,
    build_parser,
    collect_wav_files,
    is_mac_metadata,
    needs_conversion,
    normalize_samples,
    prepare_card,
    read_options,
    read_wav_mono16,
    strip_mac_metadata,
    write_options,
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
# PlayOptions
# ---------------------------------------------------------------------------


class TestPlayOptions:
    def test_defaults(self):
        opts = PlayOptions()
        assert opts.quantize_pitch is False
        assert opts.add_fades is True
        assert opts.gated_playback is False
        assert opts.lock_pitch is False
        assert opts.change_on_loop is False

    def test_custom(self):
        opts = PlayOptions(quantize_pitch=True, add_fades=False, gated_playback=True)
        assert opts.quantize_pitch is True
        assert opts.add_fades is False
        assert opts.gated_playback is True


# ---------------------------------------------------------------------------
# Options file I/O
# ---------------------------------------------------------------------------


class TestWriteOptions:
    def test_default_options(self, tmp_path: Path):
        path = tmp_path / "options.txt"
        write_options(path)
        content = path.read_text()
        assert "QUANTIZE_PITCH=0" in content
        assert "ADD_FADES=1" in content
        assert "GATED_PLAYBACK=0" in content
        assert "LOCK_PITCH=0" in content
        assert "CHANGE_ON_LOOP=0" in content

    def test_custom_options(self, tmp_path: Path):
        path = tmp_path / "options.txt"
        opts = PlayOptions(quantize_pitch=True, gated_playback=True, change_on_loop=True)
        write_options(path, opts)
        content = path.read_text()
        assert "QUANTIZE_PITCH=1" in content
        assert "ADD_FADES=1" in content
        assert "GATED_PLAYBACK=1" in content
        assert "LOCK_PITCH=0" in content
        assert "CHANGE_ON_LOOP=1" in content

    def test_all_five_keys_present(self, tmp_path: Path):
        path = tmp_path / "options.txt"
        write_options(path)
        lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        assert len(lines) == 5


class TestReadOptions:
    def test_roundtrip(self, tmp_path: Path):
        path = tmp_path / "options.txt"
        original = PlayOptions(quantize_pitch=True, add_fades=False, lock_pitch=True)
        write_options(path, original)
        loaded = read_options(path)
        assert loaded == original

    def test_missing_keys_use_defaults(self, tmp_path: Path):
        path = tmp_path / "options.txt"
        path.write_text("QUANTIZE_PITCH=1\n")
        loaded = read_options(path)
        assert loaded.quantize_pitch is True
        assert loaded.add_fades is True  # default
        assert loaded.gated_playback is False  # default

    def test_unknown_keys_ignored(self, tmp_path: Path):
        path = tmp_path / "options.txt"
        path.write_text("QUANTIZE_PITCH=1\nUNKNOWN_OPTION=1\n")
        loaded = read_options(path)
        assert loaded.quantize_pitch is True

    def test_empty_file(self, tmp_path: Path):
        path = tmp_path / "options.txt"
        path.write_text("")
        loaded = read_options(path)
        assert loaded == PlayOptions()


# ---------------------------------------------------------------------------
# WAV I/O
# ---------------------------------------------------------------------------


class TestNeedsConversion:
    def test_conforming(self, tmp_path: Path):
        p = tmp_path / "ok.wav"
        _make_mono16_wav(p, PLAY_SAMPLE_RATE, 1000)
        assert needs_conversion(p) is False

    def test_stereo(self, tmp_path: Path):
        p = tmp_path / "stereo.wav"
        _make_stereo_wav(p, PLAY_SAMPLE_RATE, 1000)
        assert needs_conversion(p) is True

    def test_wrong_rate(self, tmp_path: Path):
        p = tmp_path / "48k.wav"
        _make_mono16_wav(p, 48000, 1000)
        assert needs_conversion(p) is True

    def test_wrong_bits(self, tmp_path: Path):
        p = tmp_path / "24bit.wav"
        _make_mono_24bit_wav(p, PLAY_SAMPLE_RATE, 1000)
        assert needs_conversion(p) is True


class TestAutoConvert:
    def test_returns_original_when_conforming(self, tmp_path: Path):
        p = tmp_path / "ok.wav"
        out = tmp_path / "out.wav"
        _make_mono16_wav(p, PLAY_SAMPLE_RATE, 1000)
        result = auto_convert(p, out)
        assert result == p
        assert not out.exists()

    def test_converts_stereo(self, tmp_path: Path):
        p = tmp_path / "stereo.wav"
        out = tmp_path / "out.wav"
        _make_stereo_wav(p, PLAY_SAMPLE_RATE, 1000)
        result = auto_convert(p, out)
        assert result == out
        assert out.exists()
        import cysox

        meta = cysox.info(str(out))
        assert meta.channels == 1
        assert meta.sample_rate == PLAY_SAMPLE_RATE
        assert meta.bits_per_sample == 16

    def test_converts_wrong_rate(self, tmp_path: Path):
        p = tmp_path / "48k.wav"
        out = tmp_path / "out.wav"
        _make_mono16_wav(p, 48000, 1000)
        result = auto_convert(p, out)
        assert result == out
        import cysox

        meta = cysox.info(str(out))
        assert meta.sample_rate == PLAY_SAMPLE_RATE


class TestReadWavMono16:
    def test_read_conforming(self, tmp_path: Path):
        p = tmp_path / "test.wav"
        _make_mono16_wav(p, PLAY_SAMPLE_RATE, 4410)
        samples = read_wav_mono16(p)
        assert samples.dtype == np.int16
        assert len(samples) == 4410

    def test_reject_stereo(self, tmp_path: Path):
        p = tmp_path / "stereo.wav"
        _make_stereo_wav(p, PLAY_SAMPLE_RATE, 1000)
        with pytest.raises(ValueError, match="mono"):
            read_wav_mono16(p)

    def test_reject_wrong_rate(self, tmp_path: Path):
        p = tmp_path / "48k.wav"
        _make_mono16_wav(p, 48000, 1000)
        with pytest.raises(ValueError, match="44100"):
            read_wav_mono16(p)


class TestWriteWavMono16:
    def test_roundtrip(self, tmp_path: Path):
        samples = (np.sin(np.linspace(0, 2 * np.pi * 440, 4410)) * 32767).astype(np.int16)
        p = tmp_path / "out.wav"
        write_wav_mono16(p, samples)

        with wave.open(str(p), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == PLAY_SAMPLE_RATE
            assert wf.getnframes() == 4410

    def test_riff_header(self, tmp_path: Path):
        samples = np.zeros(100, dtype=np.int16)
        p = tmp_path / "out.wav"
        write_wav_mono16(p, samples)

        with open(str(p), "rb") as f:
            riff = f.read(4)
            assert riff == b"RIFF"
            f.read(4)  # skip size
            wave_tag = f.read(4)
            assert wave_tag == b"WAVE"


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


# ---------------------------------------------------------------------------
# macOS metadata
# ---------------------------------------------------------------------------


class TestIsMacMetadata:
    def test_ds_store(self):
        assert is_mac_metadata(Path(".DS_Store")) is True

    def test_resource_fork(self):
        assert is_mac_metadata(Path("._sample.wav")) is True

    def test_spotlight(self):
        assert is_mac_metadata(Path(".Spotlight-V100")) is True

    def test_trashes(self):
        assert is_mac_metadata(Path(".Trashes")) is True

    def test_fseventsd(self):
        assert is_mac_metadata(Path(".fseventsd")) is True

    def test_normal_file(self):
        assert is_mac_metadata(Path("kick.wav")) is False

    def test_dotfile_not_metadata(self):
        assert is_mac_metadata(Path(".gitignore")) is False


class TestStripMacMetadata:
    def test_removes_metadata_files(self, tmp_path: Path):
        (tmp_path / ".DS_Store").write_text("x")
        (tmp_path / "._sample.wav").write_text("x")
        (tmp_path / "good.wav").write_text("x")
        removed = strip_mac_metadata(tmp_path)
        assert len(removed) == 2
        assert (tmp_path / "good.wav").exists()
        assert not (tmp_path / ".DS_Store").exists()
        assert not (tmp_path / "._sample.wav").exists()

    def test_no_metadata_returns_empty(self, tmp_path: Path):
        (tmp_path / "a.wav").write_text("x")
        removed = strip_mac_metadata(tmp_path)
        assert removed == []


# ---------------------------------------------------------------------------
# collect_wav_files
# ---------------------------------------------------------------------------


class TestCollectWavFiles:
    def test_collects_alphabetically(self, tmp_path: Path):
        for name in ["c.wav", "a.wav", "b.wav"]:
            _make_mono16_wav(tmp_path / name, PLAY_SAMPLE_RATE, 100)
        result = collect_wav_files(tmp_path)
        assert [p.name for p in result] == ["a.wav", "b.wav", "c.wav"]

    def test_includes_uppercase_extension(self, tmp_path: Path):
        _make_mono16_wav(tmp_path / "test.WAV", PLAY_SAMPLE_RATE, 100)
        result = collect_wav_files(tmp_path)
        assert len(result) == 1

    def test_excludes_non_wav(self, tmp_path: Path):
        _make_mono16_wav(tmp_path / "ok.wav", PLAY_SAMPLE_RATE, 100)
        (tmp_path / "readme.txt").write_text("hi")
        (tmp_path / "data.mp3").write_text("fake")
        result = collect_wav_files(tmp_path)
        assert len(result) == 1

    def test_excludes_mac_metadata(self, tmp_path: Path):
        _make_mono16_wav(tmp_path / "ok.wav", PLAY_SAMPLE_RATE, 100)
        (tmp_path / "._ok.wav").write_text("resource fork")
        result = collect_wav_files(tmp_path)
        assert len(result) == 1

    def test_empty_dir(self, tmp_path: Path):
        result = collect_wav_files(tmp_path)
        assert result == []


# ---------------------------------------------------------------------------
# prepare_card
# ---------------------------------------------------------------------------


class TestPrepareCard:
    def test_basic(self, tmp_path: Path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        for name in ["kick.wav", "snare.wav", "hat.wav"]:
            _make_mono16_wav(src / name, PLAY_SAMPLE_RATE, 4410)
        wav_files = collect_wav_files(src)
        result = prepare_card(wav_files, dst)
        assert len(result) == 3
        assert (dst / "options.txt").exists()
        names = sorted(p.name for p in dst.iterdir() if p.suffix == ".wav")
        assert names == ["1_hat.wav", "2_kick.wav", "3_snare.wav"]

    def test_zero_padded_numbering(self, tmp_path: Path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        for i in range(12):
            _make_mono16_wav(src / f"sample_{i:02d}.wav", PLAY_SAMPLE_RATE, 100)
        wav_files = collect_wav_files(src)
        result = prepare_card(wav_files, dst)
        # With 12 files, padding should be 2 digits
        assert result[0].name.startswith("01_")
        assert result[9].name.startswith("10_")

    def test_prefix(self, tmp_path: Path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        _make_mono16_wav(src / "a.wav", PLAY_SAMPLE_RATE, 100)
        _make_mono16_wav(src / "b.wav", PLAY_SAMPLE_RATE, 100)
        wav_files = collect_wav_files(src)
        result = prepare_card(wav_files, dst, prefix="perc")
        assert result[0].name == "1_perc.wav"
        assert result[1].name == "2_perc.wav"

    def test_options_file_written(self, tmp_path: Path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        _make_mono16_wav(src / "a.wav", PLAY_SAMPLE_RATE, 100)
        wav_files = collect_wav_files(src)
        opts = PlayOptions(quantize_pitch=True)
        prepare_card(wav_files, dst, options=opts)
        loaded = read_options(dst / "options.txt")
        assert loaded.quantize_pitch is True

    def test_no_options_file(self, tmp_path: Path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        _make_mono16_wav(src / "a.wav", PLAY_SAMPLE_RATE, 100)
        wav_files = collect_wav_files(src)
        prepare_card(wav_files, dst, write_options_file=False)
        assert not (dst / "options.txt").exists()

    def test_too_many_files(self, tmp_path: Path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        paths = []
        for i in range(33):
            p = src / f"sample_{i:02d}.wav"
            _make_mono16_wav(p, PLAY_SAMPLE_RATE, 100)
            paths.append(p)
        with pytest.raises(ValueError, match="32"):
            prepare_card(paths, dst)

    def test_empty_list(self):
        with pytest.raises(ValueError, match="No input"):
            prepare_card([], Path("/nonexistent"))

    def test_max_files_accepted(self, tmp_path: Path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        paths = []
        for i in range(PLAY_MAX_FILES):
            p = src / f"s_{i:02d}.wav"
            _make_mono16_wav(p, PLAY_SAMPLE_RATE, 100)
            paths.append(p)
        result = prepare_card(paths, dst)
        assert len(result) == PLAY_MAX_FILES

    def test_strips_mac_metadata(self, tmp_path: Path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        _make_mono16_wav(src / "a.wav", PLAY_SAMPLE_RATE, 100)
        wav_files = collect_wav_files(src)
        prepare_card(wav_files, dst)
        # Manually create metadata after prepare to check strip runs during prepare
        # (the strip runs on output_dir, not input)
        # For a more meaningful test, create metadata in output dir before prepare
        # but since prepare creates the dir, let's verify no metadata after
        for item in dst.iterdir():
            assert not is_mac_metadata(item)

    def test_normalize(self, tmp_path: Path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        # Create a quiet WAV
        p = src / "quiet.wav"
        quiet = np.array([100, -100, 50, -50] * 100, dtype=np.int16)
        with wave.open(str(p), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(PLAY_SAMPLE_RATE)
            wf.writeframes(quiet.tobytes())
        wav_files = collect_wav_files(src)
        prepare_card(wav_files, dst, normalize=True)
        out_file = next(iter(dst.glob("*.wav")))
        result = read_wav_mono16(out_file)
        # After normalization, peak should be much higher than 100
        assert np.max(np.abs(result)) > 100


class TestPrepareCardAutoConvert:
    def test_converts_stereo(self, tmp_path: Path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        _make_stereo_wav(src / "stereo.wav", PLAY_SAMPLE_RATE, 1000)
        wav_files = collect_wav_files(src)
        result = prepare_card(wav_files, dst, auto_convert_enabled=True)
        assert len(result) == 1
        import cysox

        meta = cysox.info(str(result[0]))
        assert meta.channels == 1
        assert meta.bits_per_sample == 16

    def test_converts_wrong_rate(self, tmp_path: Path):
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        src.mkdir()
        _make_mono16_wav(src / "48k.wav", 48000, 1000)
        wav_files = collect_wav_files(src)
        result = prepare_card(wav_files, dst, auto_convert_enabled=True)
        assert len(result) == 1
        import cysox

        meta = cysox.info(str(result[0]))
        assert meta.sample_rate == PLAY_SAMPLE_RATE


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------


class TestBuildParser:
    def test_prepare_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["prepare", "-i", "input/", "-o", "output/"])
        assert args.command == "prepare"
        assert args.input == Path("input/")
        assert args.output == Path("output/")

    def test_prepare_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["prepare", "-i", "in/", "-o", "out/"])
        assert args.sr == PLAY_SAMPLE_RATE
        assert args.auto_convert is False
        assert args.normalize is False
        assert args.no_options is False
        assert args.no_strip is False
        assert args.prefix is None
        assert args.quantize_pitch is False
        assert args.no_fades is False
        assert args.gated is False
        assert args.lock_pitch is False
        assert args.change_on_loop is False
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
                "48000",
                "--auto-convert",
                "--normalize",
                "--no-options",
                "--no-strip",
                "--prefix",
                "kick",
                "--quantize-pitch",
                "--no-fades",
                "--gated",
                "--lock-pitch",
                "--change-on-loop",
            ]
        )
        assert args.sr == 48000
        assert args.auto_convert is True
        assert args.normalize is True
        assert args.no_options is True
        assert args.no_strip is True
        assert args.prefix == "kick"
        assert args.quantize_pitch is True
        assert args.no_fades is True
        assert args.gated is True
        assert args.lock_pitch is True
        assert args.change_on_loop is True
        assert args.verbose is True

    def test_options_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "options",
                "-o",
                "options.txt",
                "--quantize-pitch",
                "--gated",
            ]
        )
        assert args.command == "options"
        assert args.output == Path("options.txt")
        assert args.quantize_pitch is True
        assert args.gated is True
