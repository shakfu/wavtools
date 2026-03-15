import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from wavtools.octatrack import (
    OT_DEFAULT_BPM,
    OT_MAX_SLICES,
    OT_NORMALIZE_DB,
    OT_SAMPLE_RATE,
    OTSlice,
    auto_convert,
    build_chain,
    build_parser,
    generate_ot_data,
    needs_conversion,
    normalize_samples,
    read_wav_mono16,
    select_random,
    write_ot_file,
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
        # Pack as 3 bytes little-endian
        b = int(s) & 0xFFFFFF
        raw += struct.pack("<I", b)[:3]
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(3)
        wf.setframerate(sr)
        wf.writeframes(raw)


# ---------------------------------------------------------------------------
# OTSlice
# ---------------------------------------------------------------------------


class TestOTSlice:
    def test_basic(self):
        s = OTSlice(100, 500)
        assert s.start_point == 100
        assert s.length == 500
        assert s.loop_point == 500

    def test_custom_loop_point(self):
        s = OTSlice(0, 1000, loop_point=800)
        assert s.loop_point == 800


# ---------------------------------------------------------------------------
# read_wav_mono16
# ---------------------------------------------------------------------------


class TestReadWavMono16:
    def test_reads_mono_16bit(self, tmp_path):
        p = tmp_path / "mono.wav"
        _make_mono16_wav(p, OT_SAMPLE_RATE, 1000)

        samples = read_wav_mono16(p)
        assert samples.dtype == np.int16
        assert len(samples) == 1000

    def test_rejects_stereo(self, tmp_path):
        p = tmp_path / "stereo.wav"
        _make_stereo_wav(p, OT_SAMPLE_RATE, 500)

        with pytest.raises(ValueError, match="mono"):
            read_wav_mono16(p)

    def test_rejects_wrong_sample_rate(self, tmp_path):
        p = tmp_path / "wrong_sr.wav"
        _make_mono16_wav(p, 48000, 1000)

        with pytest.raises(ValueError, match="44100"):
            read_wav_mono16(p)


# ---------------------------------------------------------------------------
# write_wav_mono16
# ---------------------------------------------------------------------------


class TestWriteWavMono16:
    def test_valid_riff_header(self, tmp_path):
        p = tmp_path / "out.wav"
        write_wav_mono16(p, np.zeros(100, dtype=np.int16))

        with open(p, "rb") as f:
            assert f.read(4) == b"RIFF"
            f.read(4)
            assert f.read(4) == b"WAVE"

    def test_riff_size(self, tmp_path):
        p = tmp_path / "out.wav"
        write_wav_mono16(p, np.zeros(100, dtype=np.int16))

        file_size = p.stat().st_size
        with open(p, "rb") as f:
            f.read(4)
            riff_size = struct.unpack("<I", f.read(4))[0]
        assert riff_size == file_size - 8

    def test_mono_16bit_format(self, tmp_path):
        p = tmp_path / "out.wav"
        write_wav_mono16(p, np.zeros(100, dtype=np.int16), 44100)

        with open(p, "rb") as f:
            f.seek(20)
            fmt_code = struct.unpack("<H", f.read(2))[0]
            channels = struct.unpack("<H", f.read(2))[0]
            sr = struct.unpack("<I", f.read(4))[0]
            f.read(4)  # byte rate
            f.read(2)  # block align
            bits = struct.unpack("<H", f.read(2))[0]
        assert fmt_code == 1  # PCM
        assert channels == 1
        assert sr == 44100
        assert bits == 16

    def test_data_roundtrip(self, tmp_path):
        p = tmp_path / "rt.wav"
        original = np.array([100, -100, 32767, -32768, 0], dtype=np.int16)
        write_wav_mono16(p, original)

        with open(p, "rb") as f:
            f.seek(44)
            recovered = np.frombuffer(f.read(10), dtype=np.int16)
        np.testing.assert_array_equal(recovered, original)


# ---------------------------------------------------------------------------
# generate_ot_data
# ---------------------------------------------------------------------------


class TestGenerateOtData:
    def test_starts_with_form_header(self):
        slices = [OTSlice(0, 1000)]
        data = generate_ot_data(slices, 1000)
        assert data[:4] == b"FORM"

    def test_contains_dps1_smpa(self):
        slices = [OTSlice(0, 1000)]
        data = generate_ot_data(slices, 1000)
        assert b"DPS1" in data
        assert b"SMPA" in data

    def test_slice_count_encoded(self):
        slices = [OTSlice(0, 500), OTSlice(500, 500)]
        data = generate_ot_data(slices, 1000)
        # Slice count is a big-endian u32 near the end (before the 2-byte checksum)
        count_bytes = data[-6:-2]
        count = struct.unpack(">I", count_bytes)[0]
        assert count == 2

    def test_max_slices_accepted(self):
        slices = [OTSlice(i * 100, 100) for i in range(OT_MAX_SLICES)]
        data = generate_ot_data(slices, OT_MAX_SLICES * 100)
        count_bytes = data[-6:-2]
        count = struct.unpack(">I", count_bytes)[0]
        assert count == OT_MAX_SLICES

    def test_exceeding_max_slices_raises(self):
        slices = [OTSlice(i * 100, 100) for i in range(OT_MAX_SLICES + 1)]
        with pytest.raises(ValueError, match="64"):
            generate_ot_data(slices, (OT_MAX_SLICES + 1) * 100)

    def test_tempo_encoding(self):
        slices = [OTSlice(0, 1000)]
        data = generate_ot_data(slices, 1000, tempo=120)
        # Tempo field is at offset 22 (after 22 bytes of header), big-endian u32
        tempo_val = struct.unpack(">I", data[22:26])[0]
        assert tempo_val == 120 * 6 * 4

    def test_trim_end_equals_total_samples(self):
        total = 44100
        slices = [OTSlice(0, total)]
        data = generate_ot_data(slices, total)
        # TrimEnd is at a known offset: 22 (header) + 4 (tempo) + 4 + 4 + 4 + 4 + 2 + 1 + 4 = 49
        trim_end = struct.unpack(">I", data[49:53])[0]
        assert trim_end == total

    def test_trim_len_uses_tempo_parameter(self):
        slices = [OTSlice(0, 44100)]
        data_120 = generate_ot_data(slices, 44100, tempo=120)
        data_180 = generate_ot_data(slices, 44100, tempo=180)
        # TrimLen is at offset 26 (22 header + 4 tempo), big-endian u32
        trim_120 = struct.unpack(">I", data_120[26:30])[0]
        trim_180 = struct.unpack(">I", data_180[26:30])[0]
        # Different tempos must produce different trim lengths
        assert trim_120 != trim_180
        # Verify the actual values: trim_len = round(tempo * total / (sr * 60)) * 25
        expected_120 = int((120 * 44100 / (44100 * 60)) + 0.5) * 25  # round(2) * 25 = 50
        expected_180 = int((180 * 44100 / (44100 * 60)) + 0.5) * 25  # round(3) * 25 = 75
        assert trim_120 == expected_120
        assert trim_180 == expected_180

    def test_checksum_is_last_two_bytes(self):
        slices = [OTSlice(0, 1000)]
        data = generate_ot_data(slices, 1000)
        expected_checksum = sum(data[16:-2]) & 0xFFFF
        actual_checksum = struct.unpack(">H", data[-2:])[0]
        assert actual_checksum == expected_checksum

    def test_slice_start_end_loop_encoding(self):
        s = OTSlice(100, 500, loop_point=400)
        data = generate_ot_data([s], 600)
        # First slice entry starts at offset 57 (after all header/metadata fields)
        # 22 (header) + 4 (tempo) + 4 (trimlen) + 4 (looplen) + 4 (stretch)
        # + 4 (loop) + 2 (gain) + 1 (quantize) + 4 (trimstart) + 4 (trimend)
        # + 4 (looppoint) = 57
        slice_offset = 57
        start = struct.unpack(">I", data[slice_offset : slice_offset + 4])[0]
        end = struct.unpack(">I", data[slice_offset + 4 : slice_offset + 8])[0]
        loop = struct.unpack(">I", data[slice_offset + 8 : slice_offset + 12])[0]
        assert start == 100
        assert end == 600  # start_point + length
        assert loop == 400


# ---------------------------------------------------------------------------
# write_ot_file
# ---------------------------------------------------------------------------


class TestWriteOtFile:
    def test_writes_file(self, tmp_path):
        p = tmp_path / "test.ot"
        slices = [OTSlice(0, 1000)]
        write_ot_file(p, slices, 1000)
        assert p.exists()
        assert p.stat().st_size > 0

    def test_file_starts_with_form(self, tmp_path):
        p = tmp_path / "test.ot"
        write_ot_file(p, [OTSlice(0, 1000)], 1000)
        with open(p, "rb") as f:
            assert f.read(4) == b"FORM"


# ---------------------------------------------------------------------------
# build_chain (integration)
# ---------------------------------------------------------------------------


class TestBuildChain:
    def test_basic_chain(self, tmp_path):
        in_dir = tmp_path / "in"
        in_dir.mkdir()

        for i in range(3):
            _make_mono16_wav(in_dir / f"sample_{i}.wav", OT_SAMPLE_RATE, 1000)

        out_wav = tmp_path / "chain.wav"
        slices = build_chain(
            sorted(in_dir.glob("*.wav")),
            out_wav,
        )
        assert len(slices) == 3
        assert out_wav.exists()
        assert out_wav.with_suffix(".ot").exists()

    def test_tight_packing(self, tmp_path):
        in_dir = tmp_path / "in"
        in_dir.mkdir()

        _make_mono16_wav(in_dir / "a.wav", OT_SAMPLE_RATE, 500)
        _make_mono16_wav(in_dir / "b.wav", OT_SAMPLE_RATE, 1000)

        out_wav = tmp_path / "chain.wav"
        slices = build_chain(sorted(in_dir.glob("*.wav")), out_wav, evenly_spaced=False)

        assert slices[0].start_point == 0
        assert slices[0].length == 500
        assert slices[1].start_point == 500
        assert slices[1].length == 1000

    def test_even_spacing(self, tmp_path):
        in_dir = tmp_path / "in"
        in_dir.mkdir()

        _make_mono16_wav(in_dir / "a.wav", OT_SAMPLE_RATE, 500)
        _make_mono16_wav(in_dir / "b.wav", OT_SAMPLE_RATE, 1000)

        out_wav = tmp_path / "chain.wav"
        slices = build_chain(sorted(in_dir.glob("*.wav")), out_wav, evenly_spaced=True)

        assert slices[0].start_point == 0
        assert slices[0].length == 1000  # padded to max
        assert slices[1].start_point == 1000
        assert slices[1].length == 1000

    def test_ot_file_slice_count_matches(self, tmp_path):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        for i in range(4):
            _make_mono16_wav(in_dir / f"s{i}.wav", OT_SAMPLE_RATE, 800)

        out_wav = tmp_path / "chain.wav"
        build_chain(sorted(in_dir.glob("*.wav")), out_wav)

        ot_path = out_wav.with_suffix(".ot")
        data = ot_path.read_bytes()
        count = struct.unpack(">I", data[-6:-2])[0]
        assert count == 4

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="No input"):
            build_chain([], Path("out.wav"))

    def test_too_many_files_raises(self, tmp_path):
        paths = [tmp_path / f"s{i}.wav" for i in range(OT_MAX_SLICES + 1)]
        with pytest.raises(ValueError, match="64"):
            build_chain(paths, tmp_path / "out.wav")

    def test_chain_wav_sample_count(self, tmp_path):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _make_mono16_wav(in_dir / "a.wav", OT_SAMPLE_RATE, 500)
        _make_mono16_wav(in_dir / "b.wav", OT_SAMPLE_RATE, 700)

        out_wav = tmp_path / "chain.wav"
        build_chain(sorted(in_dir.glob("*.wav")), out_wav, evenly_spaced=False)

        # Read the output WAV and verify total sample count
        with wave.open(str(out_wav), "rb") as wf:
            assert wf.getnframes() == 1200
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2


# ---------------------------------------------------------------------------
# needs_conversion / auto_convert
# ---------------------------------------------------------------------------


class TestNeedsConversion:
    def test_conforming_file(self, tmp_path):
        p = tmp_path / "ok.wav"
        _make_mono16_wav(p, OT_SAMPLE_RATE, 1000)
        assert needs_conversion(p) is False

    def test_stereo_needs_conversion(self, tmp_path):
        p = tmp_path / "stereo.wav"
        _make_stereo_wav(p, OT_SAMPLE_RATE, 500)
        assert needs_conversion(p) is True

    def test_wrong_rate_needs_conversion(self, tmp_path):
        p = tmp_path / "48k.wav"
        _make_mono16_wav(p, 48000, 1000)
        assert needs_conversion(p) is True

    def test_wrong_bits_needs_conversion(self, tmp_path):
        p = tmp_path / "24bit.wav"
        _make_mono_24bit_wav(p, OT_SAMPLE_RATE, 1000)
        assert needs_conversion(p) is True


class TestAutoConvert:
    def test_returns_original_when_conforming(self, tmp_path):
        p = tmp_path / "ok.wav"
        _make_mono16_wav(p, OT_SAMPLE_RATE, 1000)
        result = auto_convert(p, tmp_path / "out.wav")
        assert result == p
        assert not (tmp_path / "out.wav").exists()

    def test_converts_stereo_to_mono(self, tmp_path):
        p = tmp_path / "stereo.wav"
        _make_stereo_wav(p, OT_SAMPLE_RATE, 1000)
        out = tmp_path / "converted.wav"
        result = auto_convert(p, out)
        assert result == out
        assert out.exists()
        import cysox

        meta = cysox.info(str(out))
        assert meta.channels == 1
        assert meta.sample_rate == OT_SAMPLE_RATE

    def test_converts_wrong_sample_rate(self, tmp_path):
        p = tmp_path / "48k.wav"
        _make_mono16_wav(p, 48000, 4800)
        out = tmp_path / "converted.wav"
        result = auto_convert(p, out)
        assert result == out
        import cysox

        meta = cysox.info(str(out))
        assert meta.sample_rate == OT_SAMPLE_RATE


# ---------------------------------------------------------------------------
# normalize_samples
# ---------------------------------------------------------------------------


class TestNormalizeSamples:
    def test_quiet_signal_boosted(self):
        samples = np.array([100, -100, 50, -50], dtype=np.int16)
        result = normalize_samples(samples)
        peak = np.max(np.abs(result.astype(np.float64)))
        target_peak = 32767.0 * (10.0 ** (OT_NORMALIZE_DB / 20.0))
        assert abs(peak - target_peak) < 2  # within rounding

    def test_already_loud_signal(self):
        samples = np.array([32767, -32768, 0, 16000], dtype=np.int16)
        result = normalize_samples(samples)
        peak = np.max(np.abs(result.astype(np.float64)))
        target_peak = 32767.0 * (10.0 ** (OT_NORMALIZE_DB / 20.0))
        assert abs(peak - target_peak) < 2

    def test_silent_signal_unchanged(self):
        samples = np.zeros(100, dtype=np.int16)
        result = normalize_samples(samples)
        np.testing.assert_array_equal(result, samples)

    def test_preserves_dtype(self):
        samples = np.array([1000, -1000], dtype=np.int16)
        result = normalize_samples(samples)
        assert result.dtype == np.int16


# ---------------------------------------------------------------------------
# select_random
# ---------------------------------------------------------------------------


class TestSelectRandom:
    def test_fewer_than_count(self):
        paths = [Path(f"s{i}.wav") for i in range(5)]
        result = select_random(paths, 64)
        assert result == paths

    def test_selects_exact_count(self):
        paths = [Path(f"s{i}.wav") for i in range(100)]
        result = select_random(paths, 10, seed=42)
        assert len(result) == 10

    def test_preserves_order(self):
        paths = [Path(f"s{i:03d}.wav") for i in range(100)]
        result = select_random(paths, 10, seed=42)
        # Result should be in the same relative order as input
        indices = [paths.index(p) for p in result]
        assert indices == sorted(indices)

    def test_deterministic_with_seed(self):
        paths = [Path(f"s{i}.wav") for i in range(100)]
        r1 = select_random(paths, 10, seed=123)
        r2 = select_random(paths, 10, seed=123)
        assert r1 == r2

    def test_different_seeds_differ(self):
        paths = [Path(f"s{i}.wav") for i in range(100)]
        r1 = select_random(paths, 10, seed=1)
        r2 = select_random(paths, 10, seed=2)
        assert r1 != r2


# ---------------------------------------------------------------------------
# build_chain with new features
# ---------------------------------------------------------------------------


class TestBuildChainAutoConvert:
    def test_converts_stereo_input(self, tmp_path):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _make_stereo_wav(in_dir / "stereo.wav", OT_SAMPLE_RATE, 1000)
        _make_mono16_wav(in_dir / "mono.wav", OT_SAMPLE_RATE, 1000)

        out_wav = tmp_path / "chain.wav"
        slices = build_chain(
            sorted(in_dir.glob("*.wav")),
            out_wav,
            auto_convert_enabled=True,
        )
        assert len(slices) == 2
        assert out_wav.exists()

    def test_converts_wrong_rate_input(self, tmp_path):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _make_mono16_wav(in_dir / "48k.wav", 48000, 4800)

        out_wav = tmp_path / "chain.wav"
        slices = build_chain(
            [in_dir / "48k.wav"],
            out_wav,
            auto_convert_enabled=True,
        )
        assert len(slices) == 1

    def test_rejects_stereo_without_auto_convert(self, tmp_path):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _make_stereo_wav(in_dir / "stereo.wav", OT_SAMPLE_RATE, 1000)

        out_wav = tmp_path / "chain.wav"
        with pytest.raises(ValueError, match="mono"):
            build_chain([in_dir / "stereo.wav"], out_wav)


class TestBuildChainNormalize:
    def test_normalize_equalizes_levels(self, tmp_path):
        in_dir = tmp_path / "in"
        in_dir.mkdir()

        # Quiet sample
        quiet = (np.sin(np.linspace(0, 2 * np.pi * 440, 1000)) * 1000).astype(np.int16)
        with wave.open(str(in_dir / "quiet.wav"), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(OT_SAMPLE_RATE)
            wf.writeframes(quiet.tobytes())

        # Loud sample
        loud = (np.sin(np.linspace(0, 2 * np.pi * 440, 1000)) * 32000).astype(np.int16)
        with wave.open(str(in_dir / "loud.wav"), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(OT_SAMPLE_RATE)
            wf.writeframes(loud.tobytes())

        out_wav = tmp_path / "chain.wav"
        build_chain(sorted(in_dir.glob("*.wav")), out_wav, normalize=True)

        # Read back the chain and check that both slices have similar peak levels
        with wave.open(str(out_wav), "rb") as wf:
            raw = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        peak1 = np.max(np.abs(raw[:1000].astype(np.float64)))
        peak2 = np.max(np.abs(raw[1000:].astype(np.float64)))
        # Both should be near the target peak
        target_peak = 32767.0 * (10.0 ** (OT_NORMALIZE_DB / 20.0))
        assert abs(peak1 - target_peak) < 2
        assert abs(peak2 - target_peak) < 2


class TestBuildChainRandomSelect:
    def test_selects_subset(self, tmp_path):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        for i in range(100):
            _make_mono16_wav(in_dir / f"s{i:03d}.wav", OT_SAMPLE_RATE, 500)

        out_wav = tmp_path / "chain.wav"
        slices = build_chain(
            sorted(in_dir.glob("*.wav")),
            out_wav,
            random_select=True,
            random_seed=42,
        )
        assert len(slices) == OT_MAX_SLICES

    def test_deterministic_with_seed(self, tmp_path):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        for i in range(80):
            _make_mono16_wav(in_dir / f"s{i:03d}.wav", OT_SAMPLE_RATE, 500)

        wav_files = sorted(in_dir.glob("*.wav"))

        out1 = tmp_path / "chain1.wav"
        s1 = build_chain(wav_files, out1, random_select=True, random_seed=99)

        out2 = tmp_path / "chain2.wav"
        s2 = build_chain(wav_files, out2, random_select=True, random_seed=99)

        assert len(s1) == len(s2)
        for a, b in zip(s1, s2, strict=True):
            assert a.start_point == b.start_point
            assert a.length == b.length


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------


class TestBuildParser:
    def test_chain_subcommand(self):
        args = build_parser().parse_args(["chain", "-i", "samples/", "-o", "out.wav"])
        assert args.command == "chain"
        assert args.input == Path("samples/")
        assert args.output == Path("out.wav")

    def test_even_flag(self):
        args = build_parser().parse_args(["chain", "-i", "in/", "-o", "o.wav", "--even"])
        assert args.even is True

    def test_bpm_option(self):
        args = build_parser().parse_args(["chain", "-i", "in/", "-o", "o.wav", "--bpm", "140"])
        assert args.bpm == 140

    def test_sr_option(self):
        args = build_parser().parse_args(["chain", "-i", "in/", "-o", "o.wav", "--sr", "48000"])
        assert args.sr == 48000

    def test_auto_convert_flag(self):
        args = build_parser().parse_args(["chain", "-i", "in/", "-o", "o.wav", "--auto-convert"])
        assert args.auto_convert is True

    def test_normalize_flag(self):
        args = build_parser().parse_args(["chain", "-i", "in/", "-o", "o.wav", "--normalize"])
        assert args.normalize is True

    def test_random_flag(self):
        args = build_parser().parse_args(["chain", "-i", "in/", "-o", "o.wav", "--random"])
        assert args.random is True

    def test_seed_option(self):
        args = build_parser().parse_args(
            ["chain", "-i", "in/", "-o", "o.wav", "--random", "--seed", "42"]
        )
        assert args.seed == 42

    def test_defaults(self):
        args = build_parser().parse_args(["chain", "-i", "in/", "-o", "o.wav"])
        assert args.bpm == OT_DEFAULT_BPM
        assert args.sr == OT_SAMPLE_RATE
        assert args.even is False
        assert args.auto_convert is False
        assert args.normalize is False
        assert args.random is False
        assert args.seed is None

    def test_verbose_flag(self):
        args = build_parser().parse_args(["-v", "chain", "-i", "in/", "-o", "o.wav"])
        assert args.verbose is True

    def test_missing_subcommand_exits(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])
