import gzip
import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from wavtools.morphagene import (
    MORPHAGENE_SAMPLE_RATE,
    build_parser,
    detect_onsets,
    load_ableton_markers,
    make_reel,
    read_wav,
    resample,
    resample_file,
    select_markers,
    write_wav,
)

# ---------------------------------------------------------------------------
# Helpers -- create test WAV files without scipy
# ---------------------------------------------------------------------------


def _write_pcm16_wav(path: Path, sr: int, channels: int, n_frames: int) -> None:
    """Create a 16-bit PCM WAV with a sine tone via the stdlib wave module."""
    tone = (np.sin(np.linspace(0, 2 * np.pi * 440, n_frames)) * 32767).astype(np.int16)
    if channels > 1:
        tone = np.column_stack([tone] * channels).flatten()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(tone.tobytes())


def _make_als(path: Path, bpm: float, locator_times: list[float]) -> None:
    """Create a minimal gzipped Ableton .als with Tempo and Locators."""
    locators = "".join(
        f'<Locator Id="{i}"><Time Value="{t}" /></Locator>' for i, t in enumerate(locator_times)
    )
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Ableton><LiveSet>"
        "<MasterTrack><Mixer>"
        f'<Tempo><Manual Value="{bpm}" /></Tempo>'
        "</Mixer></MasterTrack>"
        f"<Locators><Locators>{locators}</Locators></Locators>"
        "</LiveSet></Ableton>"
    )
    with gzip.open(str(path), "wb") as f:
        f.write(xml.encode("utf-8"))


# ---------------------------------------------------------------------------
# read_wav
# ---------------------------------------------------------------------------


class TestReadWav:
    def test_16bit_mono(self, tmp_path):
        p = tmp_path / "m16.wav"
        _write_pcm16_wav(p, 44100, 1, 1000)

        audio, sr = read_wav(p)
        assert sr == 44100
        assert audio.shape == (1, 1000)
        assert audio.dtype == np.float32
        assert np.all(np.abs(audio) <= 1.0)

    def test_16bit_stereo(self, tmp_path):
        p = tmp_path / "s16.wav"
        _write_pcm16_wav(p, 48000, 2, 500)

        audio, sr = read_wav(p)
        assert sr == 48000
        assert audio.shape == (2, 500)

    def test_float32_roundtrip(self, tmp_path):
        p = tmp_path / "f32.wav"
        original = np.array([[0.5, -0.5, 0.25, -0.25]], dtype=np.float32)
        write_wav(p, original, 44100)

        audio, sr = read_wav(p)
        assert sr == 44100
        assert audio.dtype == np.float32
        np.testing.assert_array_almost_equal(audio, original, decimal=4)

    def test_float32_precision_preserved(self, tmp_path):
        """Float32 WAVs should be read directly, not quantized through int32."""
        p = tmp_path / "precise.wav"
        # Use values that would lose precision if round-tripped through int32
        original = np.array([[1.0e-7, -1.0e-7, 3.14e-5]], dtype=np.float32)
        write_wav(p, original, 48000)

        audio, _ = read_wav(p)
        np.testing.assert_array_equal(audio, original)


# ---------------------------------------------------------------------------
# write_wav
# ---------------------------------------------------------------------------


class TestWriteWav:
    def test_valid_riff_header(self, tmp_path):
        p = tmp_path / "out.wav"
        write_wav(p, np.zeros((1, 100), dtype=np.float32), 48000)

        with open(p, "rb") as f:
            assert f.read(4) == b"RIFF"
            f.read(4)
            assert f.read(4) == b"WAVE"

    def test_riff_size_patched(self, tmp_path):
        p = tmp_path / "out.wav"
        write_wav(p, np.zeros((1, 100), dtype=np.float32), 48000)

        file_size = p.stat().st_size
        with open(p, "rb") as f:
            f.read(4)
            riff_size = struct.unpack("<I", f.read(4))[0]
        assert riff_size == file_size - 8

    def test_riff_size_includes_markers(self, tmp_path):
        p = tmp_path / "m.wav"
        markers = [{"position": 10, "label": "a"}, {"position": 20, "label": "b"}]
        write_wav(p, np.zeros((1, 100), dtype=np.float32), 48000, markers=markers)

        file_size = p.stat().st_size
        with open(p, "rb") as f:
            f.read(4)
            riff_size = struct.unpack("<I", f.read(4))[0]
        assert riff_size == file_size - 8

    def test_stereo_channel_count(self, tmp_path):
        p = tmp_path / "stereo.wav"
        write_wav(p, np.zeros((2, 200), dtype=np.float32), 44100)

        with open(p, "rb") as f:
            f.seek(22)
            channels = struct.unpack("<H", f.read(2))[0]
        assert channels == 2

    def test_cue_and_label_chunks(self, tmp_path):
        p = tmp_path / "cue.wav"
        markers = [
            {"position": 0, "label": "start"},
            {"position": 24000, "label": "mid"},
        ]
        write_wav(p, np.zeros((1, 48000), dtype=np.float32), 48000, markers=markers)

        raw = p.read_bytes()
        assert b"cue " in raw
        assert b"LIST" in raw
        assert b"adtl" in raw
        assert b"labl" in raw

    def test_no_marker_chunks_without_markers(self, tmp_path):
        p = tmp_path / "no.wav"
        write_wav(p, np.zeros((1, 100), dtype=np.float32), 48000)

        raw = p.read_bytes()
        assert b"cue " not in raw

    def test_audio_data_roundtrip(self, tmp_path):
        p = tmp_path / "rt.wav"
        audio = np.array([[0.5, -0.5, 0.25, -0.25]], dtype=np.float32)
        write_wav(p, audio, 48000)

        with open(p, "rb") as f:
            f.seek(44)
            recovered = np.frombuffer(f.read(16), dtype=np.float32)
        np.testing.assert_array_almost_equal(recovered, audio.flatten())


# ---------------------------------------------------------------------------
# resample
# ---------------------------------------------------------------------------


class TestResample:
    def test_same_rate_noop(self):
        audio = np.random.randn(1, 1000).astype(np.float32)
        result = resample(audio, 44100, 44100)
        np.testing.assert_array_equal(result, audio)

    def test_upsample_shape(self):
        audio = np.random.randn(1, 1000).astype(np.float32)
        result = resample(audio, 44100, 48000)
        assert result.shape == (1, int(1000 * 48000 / 44100))

    def test_downsample_shape(self):
        audio = np.random.randn(2, 4800).astype(np.float32)
        result = resample(audio, 48000, 44100)
        assert result.shape == (2, int(4800 * 44100 / 48000))

    def test_preserves_channel_count(self):
        audio = np.random.randn(3, 1000).astype(np.float32)
        result = resample(audio, 44100, 48000)
        assert result.shape[0] == 3


# ---------------------------------------------------------------------------
# resample_file (libsox polyphase resampler)
# ---------------------------------------------------------------------------


class TestResampleFile:
    def test_changes_sample_rate(self, tmp_path):
        src = tmp_path / "44k.wav"
        dst = tmp_path / "48k.wav"
        _write_pcm16_wav(src, 44100, 1, 44100)

        resample_file(src, dst, 48000)

        _audio, sr = read_wav(dst)
        assert sr == 48000

    def test_preserves_channels(self, tmp_path):
        src = tmp_path / "stereo.wav"
        dst = tmp_path / "resampled.wav"
        _write_pcm16_wav(src, 44100, 2, 4410)

        resample_file(src, dst, 48000)

        audio, sr = read_wav(dst)
        assert audio.shape[0] == 2
        assert sr == 48000

    def test_duration_preserved(self, tmp_path):
        src = tmp_path / "src.wav"
        dst = tmp_path / "dst.wav"
        n_frames = 44100  # exactly 1 second
        _write_pcm16_wav(src, 44100, 1, n_frames)

        resample_file(src, dst, 48000)

        audio, sr = read_wav(dst)
        duration = audio.shape[1] / sr
        assert abs(duration - 1.0) < 0.01


# ---------------------------------------------------------------------------
# load_ableton_markers
# ---------------------------------------------------------------------------


class TestLoadAbletonMarkers:
    def test_basic(self, tmp_path):
        p = tmp_path / "test.als"
        bpm, times = 120.0, [0.0, 4.0, 8.0]
        _make_als(p, bpm, times)

        result = load_ableton_markers(p)
        bps = bpm / 60.0
        expected = np.array([t / bps for t in times])
        assert len(result) == 3
        np.testing.assert_array_almost_equal(result, expected)

    def test_single_locator(self, tmp_path):
        p = tmp_path / "one.als"
        _make_als(p, 140.0, [2.0])

        result = load_ableton_markers(p)
        assert len(result) == 1
        np.testing.assert_almost_equal(result[0], 2.0 / (140.0 / 60.0))

    def test_no_locators(self, tmp_path):
        p = tmp_path / "empty.als"
        _make_als(p, 120.0, [])

        result = load_ableton_markers(p)
        assert len(result) == 0

    def test_missing_bpm_raises(self, tmp_path):
        p = tmp_path / "bad.als"
        xml = '<?xml version="1.0"?><Ableton><LiveSet></LiveSet></Ableton>'
        with gzip.open(str(p), "wb") as f:
            f.write(xml.encode("utf-8"))

        with pytest.raises(ValueError, match="No BPM"):
            load_ableton_markers(p)


# ---------------------------------------------------------------------------
# detect_onsets (cysox)
# ---------------------------------------------------------------------------


class TestDetectOnsets:
    def test_returns_sorted_float_array(self, tmp_path):
        p = tmp_path / "click.wav"
        # Two loud clicks separated by silence -- should produce at least 1 onset
        sr = 44100
        silence = np.zeros(sr // 2, dtype=np.float32)
        click = np.random.randn(200).astype(np.float32) * 0.9
        audio = np.concatenate([click, silence, click, silence])[np.newaxis, :]
        write_wav(p, audio, sr)

        onsets = detect_onsets(p)
        assert isinstance(onsets, np.ndarray)
        assert onsets.dtype == np.float64
        # onsets should be sorted
        assert np.all(np.diff(onsets) >= 0)

    def test_method_parameter_accepted(self, tmp_path):
        p = tmp_path / "tone.wav"
        _write_pcm16_wav(p, 44100, 1, 44100)

        for method in ("hfc", "flux", "energy", "complex", "superflux"):
            result = detect_onsets(p, method=method)
            assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# select_markers
# ---------------------------------------------------------------------------


class TestSelectMarkers:
    def test_fewer_than_available(self):
        onsets = np.linspace(0, 4, 8)
        result = select_markers(onsets, 4)
        assert len(result) == 4
        assert result[0] == 0.0

    def test_count_exceeds_onsets(self):
        onsets = np.array([0.0, 1.0, 2.0])
        result = select_markers(onsets, 10)
        np.testing.assert_array_equal(result, onsets)

    def test_equal_count(self):
        onsets = np.array([0.0, 1.0, 2.0])
        result = select_markers(onsets, 3)
        np.testing.assert_array_equal(result, onsets)

    def test_first_marker_forced_zero(self):
        onsets = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        result = select_markers(onsets, 3)
        assert result[0] == 0.0


# ---------------------------------------------------------------------------
# make_reel (integration)
# ---------------------------------------------------------------------------


class TestMakeReel:
    def test_full_pipeline_at_48k(self, tmp_path):
        wav_in = tmp_path / "in.wav"
        wav_out = tmp_path / "out.wav"
        _write_pcm16_wav(wav_in, MORPHAGENE_SAMPLE_RATE, 1, 48000)

        markers = np.array([0.0, 0.5])
        make_reel(wav_in, wav_out, markers)

        assert wav_out.exists()
        raw = wav_out.read_bytes()
        assert b"cue " in raw

    def test_resamples_non_48k(self, tmp_path):
        wav_in = tmp_path / "in.wav"
        wav_out = tmp_path / "out.wav"
        _write_pcm16_wav(wav_in, 44100, 1, 44100)

        make_reel(wav_in, wav_out, np.array([0.0]))

        with open(wav_out, "rb") as f:
            f.seek(24)
            written_sr = struct.unpack("<I", f.read(4))[0]
        assert written_sr == MORPHAGENE_SAMPLE_RATE

    def test_stereo_passthrough(self, tmp_path):
        wav_in = tmp_path / "stereo.wav"
        wav_out = tmp_path / "out.wav"
        _write_pcm16_wav(wav_in, 48000, 2, 4800)

        make_reel(wav_in, wav_out, np.array([0.0]))

        with open(wav_out, "rb") as f:
            f.seek(22)
            ch = struct.unpack("<H", f.read(2))[0]
        assert ch == 2


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------


class TestBuildParser:
    def test_ableton_subcommand(self):
        args = build_parser().parse_args(
            ["ableton", "-w", "in.wav", "-l", "proj.als", "-o", "out.wav"]
        )
        assert args.command == "ableton"
        assert args.wav == Path("in.wav")
        assert args.labels == Path("proj.als")
        assert args.output == Path("out.wav")

    def test_onset_subcommand(self):
        args = build_parser().parse_args(["onset", "-w", "in.wav", "-o", "out.wav", "-s", "50"])
        assert args.command == "onset"
        assert args.wav == Path("in.wav")
        assert args.output == Path("out.wav")
        assert args.splices == 50

    def test_onset_default_splices_is_none(self):
        args = build_parser().parse_args(["onset", "-w", "in.wav", "-o", "out.wav"])
        assert args.splices is None

    def test_onset_threshold_and_method(self):
        args = build_parser().parse_args(
            ["onset", "-w", "i.wav", "-o", "o.wav", "-t", "0.5", "-m", "flux"]
        )
        assert args.threshold == 0.5
        assert args.method == "flux"

    def test_onset_sensitivity(self):
        args = build_parser().parse_args(
            ["onset", "-w", "i.wav", "-o", "o.wav", "--sensitivity", "2.0"]
        )
        assert args.sensitivity == 2.0

    def test_onset_defaults(self):
        args = build_parser().parse_args(["onset", "-w", "i.wav", "-o", "o.wav"])
        assert args.threshold == 0.3
        assert args.sensitivity == 1.5
        assert args.method == "hfc"

    def test_verbose_flag(self):
        args = build_parser().parse_args(["-v", "onset", "-w", "i.wav", "-o", "o.wav"])
        assert args.verbose is True

    def test_missing_subcommand_exits(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])
