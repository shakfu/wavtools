import wave
from pathlib import Path

import numpy as np
import pytest

from wavtools.morphagene import write_wav
from wavtools.slicer import (
    slice_file_by_bpm,
    slice_file_by_count,
    slice_file_by_onsets,
    slice_points_by_bpm,
    slice_points_by_count,
    slice_points_by_onsets,
    split_file_by_silence,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pcm16_wav(path: Path, sr: int, n_frames: int) -> None:
    """Create a mono 16-bit PCM WAV with a sine tone."""
    tone = (np.sin(np.linspace(0, 2 * np.pi * 440, n_frames)) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(tone.tobytes())


def _make_clicks(path: Path, sr: int) -> None:
    """Create a WAV with loud clicks separated by silence for onset detection."""
    silence = np.zeros(sr // 2, dtype=np.float32)
    click = np.random.RandomState(42).randn(200).astype(np.float32) * 0.9
    audio = np.concatenate([click, silence, click, silence, click])[np.newaxis, :]
    write_wav(path, audio, sr)


def _make_segments_with_silence(path: Path, sr: int) -> None:
    """Create a WAV with content segments separated by silence for split testing."""
    seg_samples = int(sr * 0.5)  # 500ms segments
    gap_samples = int(sr * 0.5)  # 500ms gaps
    content = (np.sin(np.linspace(0, 2 * np.pi * 440, seg_samples)) * 32767).astype(np.int16)
    silence = np.zeros(gap_samples, dtype=np.int16)
    raw = np.concatenate([content, silence, content, silence, content])
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(raw.tobytes())


# ---------------------------------------------------------------------------
# slice_points_by_onsets
# ---------------------------------------------------------------------------


class TestSlicePointsByOnsets:
    def test_returns_sorted_array(self, tmp_path: Path) -> None:
        p = tmp_path / "clicks.wav"
        _make_clicks(p, 44100)
        points = slice_points_by_onsets(p)
        assert isinstance(points, np.ndarray)
        assert points.dtype == np.float64
        assert np.all(np.diff(points) >= 0)

    def test_detects_at_least_one(self, tmp_path: Path) -> None:
        p = tmp_path / "clicks.wav"
        _make_clicks(p, 44100)
        points = slice_points_by_onsets(p)
        assert len(points) >= 1

    def test_method_parameter(self, tmp_path: Path) -> None:
        p = tmp_path / "tone.wav"
        _make_pcm16_wav(p, 44100, 44100)
        for method in ("hfc", "flux", "energy"):
            result = slice_points_by_onsets(p, method=method)
            assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# slice_points_by_bpm
# ---------------------------------------------------------------------------


class TestSlicePointsByBpm:
    def test_basic(self, tmp_path: Path) -> None:
        p = tmp_path / "tone.wav"
        _make_pcm16_wav(p, 44100, 44100 * 2)  # 2 seconds
        points = slice_points_by_bpm(p, bpm=120.0)
        # 120 BPM = 0.5s per beat, 2s file -> 4 beats
        assert len(points) == 4
        np.testing.assert_almost_equal(points[0], 0.0)
        np.testing.assert_almost_equal(points[1], 0.5)

    def test_beats_per_slice(self, tmp_path: Path) -> None:
        p = tmp_path / "tone.wav"
        _make_pcm16_wav(p, 44100, 44100 * 4)  # 4 seconds
        points = slice_points_by_bpm(p, bpm=120.0, beats_per_slice=2)
        # 2 beats per slice at 120 BPM = 1.0s per slice, 4s file -> 4 slices
        assert len(points) == 4
        np.testing.assert_almost_equal(points[1], 1.0)

    def test_starts_at_zero(self, tmp_path: Path) -> None:
        p = tmp_path / "tone.wav"
        _make_pcm16_wav(p, 44100, 44100)
        points = slice_points_by_bpm(p, bpm=60.0)
        assert points[0] == 0.0


# ---------------------------------------------------------------------------
# slice_points_by_count
# ---------------------------------------------------------------------------


class TestSlicePointsByCount:
    def test_basic(self, tmp_path: Path) -> None:
        p = tmp_path / "tone.wav"
        _make_pcm16_wav(p, 44100, 44100)  # 1 second
        points = slice_points_by_count(p, 4)
        assert len(points) == 4
        np.testing.assert_almost_equal(points[0], 0.0)

    def test_evenly_spaced(self, tmp_path: Path) -> None:
        p = tmp_path / "tone.wav"
        _make_pcm16_wav(p, 44100, 44100 * 2)  # 2 seconds
        points = slice_points_by_count(p, 4)
        diffs = np.diff(points)
        np.testing.assert_array_almost_equal(diffs, diffs[0])

    def test_count_1(self, tmp_path: Path) -> None:
        p = tmp_path / "tone.wav"
        _make_pcm16_wav(p, 44100, 44100)
        points = slice_points_by_count(p, 1)
        assert len(points) == 1
        assert points[0] == 0.0

    def test_invalid_count_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "tone.wav"
        _make_pcm16_wav(p, 44100, 44100)
        with pytest.raises(ValueError, match="count"):
            slice_points_by_count(p, 0)


# ---------------------------------------------------------------------------
# slice_file_by_count
# ---------------------------------------------------------------------------


class TestSliceFileByCount:
    def test_creates_files(self, tmp_path: Path) -> None:
        p = tmp_path / "tone.wav"
        out = tmp_path / "slices"
        _make_pcm16_wav(p, 44100, 44100)
        paths = slice_file_by_count(p, out, 4)
        assert len(paths) == 4
        for sp in paths:
            assert sp.exists()

    def test_invalid_count_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "tone.wav"
        _make_pcm16_wav(p, 44100, 44100)
        with pytest.raises(ValueError, match="count"):
            slice_file_by_count(p, tmp_path / "out", 0)


# ---------------------------------------------------------------------------
# slice_file_by_bpm
# ---------------------------------------------------------------------------


class TestSliceFileByBpm:
    def test_creates_files(self, tmp_path: Path) -> None:
        p = tmp_path / "tone.wav"
        out = tmp_path / "slices"
        _make_pcm16_wav(p, 44100, 44100 * 2)  # 2 seconds
        paths = slice_file_by_bpm(p, out, bpm=120.0)
        assert len(paths) >= 2
        for sp in paths:
            assert sp.exists()


# ---------------------------------------------------------------------------
# slice_file_by_onsets
# ---------------------------------------------------------------------------


class TestSliceFileByOnsets:
    def test_creates_files(self, tmp_path: Path) -> None:
        p = tmp_path / "clicks.wav"
        out = tmp_path / "slices"
        _make_clicks(p, 44100)
        paths = slice_file_by_onsets(p, out)
        assert len(paths) >= 1
        for sp in paths:
            assert sp.exists()


# ---------------------------------------------------------------------------
# split_file_by_silence
# ---------------------------------------------------------------------------


class TestSplitFileBySilence:
    def test_splits_segments(self, tmp_path: Path) -> None:
        p = tmp_path / "multi.wav"
        out = tmp_path / "segments"
        _make_segments_with_silence(p, 44100)
        paths = split_file_by_silence(p, out)
        assert len(paths) >= 2
        for sp in paths:
            assert sp.exists()


# ---------------------------------------------------------------------------
# Integration: slicer -> morphagene reel
# ---------------------------------------------------------------------------


class TestSlicerMorphageneIntegration:
    def test_bpm_markers_to_reel(self, tmp_path: Path) -> None:
        from wavtools.morphagene import MORPHAGENE_SAMPLE_RATE, make_reel

        wav_in = tmp_path / "in.wav"
        wav_out = tmp_path / "reel.wav"
        _make_pcm16_wav(wav_in, MORPHAGENE_SAMPLE_RATE, MORPHAGENE_SAMPLE_RATE * 2)

        markers = slice_points_by_bpm(wav_in, bpm=120.0)
        make_reel(wav_in, wav_out, markers)

        assert wav_out.exists()
        raw = wav_out.read_bytes()
        assert b"cue " in raw

    def test_count_markers_to_reel(self, tmp_path: Path) -> None:
        from wavtools.morphagene import MORPHAGENE_SAMPLE_RATE, make_reel

        wav_in = tmp_path / "in.wav"
        wav_out = tmp_path / "reel.wav"
        _make_pcm16_wav(wav_in, MORPHAGENE_SAMPLE_RATE, MORPHAGENE_SAMPLE_RATE)

        markers = slice_points_by_count(wav_in, 8)
        make_reel(wav_in, wav_out, markers)

        assert wav_out.exists()
        raw = wav_out.read_bytes()
        assert b"cue " in raw


# ---------------------------------------------------------------------------
# CLI parser extensions
# ---------------------------------------------------------------------------


class TestMorphageneSliceParser:
    def test_slice_bpm(self) -> None:
        from wavtools.morphagene import build_parser

        args = build_parser().parse_args(["slice", "-w", "in.wav", "-o", "out.wav", "--bpm", "120"])
        assert args.command == "slice"
        assert args.bpm == 120.0
        assert args.count is None

    def test_slice_count(self) -> None:
        from wavtools.morphagene import build_parser

        args = build_parser().parse_args(["slice", "-w", "in.wav", "-o", "out.wav", "-n", "16"])
        assert args.count == 16
        assert args.bpm is None

    def test_slice_onset_defaults(self) -> None:
        from wavtools.morphagene import build_parser

        args = build_parser().parse_args(["slice", "-w", "in.wav", "-o", "out.wav"])
        assert args.bpm is None
        assert args.count is None
        assert args.threshold == 0.3
        assert args.method == "hfc"

    def test_slice_splices_limit(self) -> None:
        from wavtools.morphagene import build_parser

        args = build_parser().parse_args(
            ["slice", "-w", "in.wav", "-o", "out.wav", "--bpm", "120", "-s", "50"]
        )
        assert args.splices == 50


class TestOctatrackSliceParser:
    def test_slice_bpm(self) -> None:
        from wavtools.octatrack import build_parser

        args = build_parser().parse_args(["slice", "-i", "in.wav", "-o", "out.wav", "--bpm", "120"])
        assert args.command == "slice"
        assert args.bpm == 120.0

    def test_slice_count(self) -> None:
        from wavtools.octatrack import build_parser

        args = build_parser().parse_args(["slice", "-i", "in.wav", "-o", "out.wav", "-n", "8"])
        assert args.count == 8

    def test_slice_split(self) -> None:
        from wavtools.octatrack import build_parser

        args = build_parser().parse_args(["slice", "-i", "in.wav", "-o", "out.wav", "--split"])
        assert args.split is True

    def test_slice_onset_defaults(self) -> None:
        from wavtools.octatrack import build_parser

        args = build_parser().parse_args(["slice", "-i", "in.wav", "-o", "out.wav"])
        assert args.bpm is None
        assert args.count is None
        assert args.split is False
        assert args.threshold == 0.3
        assert args.method == "hfc"
        assert args.tempo == 124
        assert args.sr == 44100
