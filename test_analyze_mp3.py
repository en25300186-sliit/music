"""Tests for analyze_mp3.py

Tests cover:
- freq_to_note: frequency-to-musical-note conversion
- find_dominant_frequencies: FFT-based peak detection on synthetic signals
- classify_instrument: instrument classification from spectral features
- detect_note_durations: grouping of frames into note events
- analyze_frame: per-frame analysis on synthetic audio
- analyze_mp3: end-to-end analysis on a synthetic WAV written to a temp file
"""

import math
import os
import struct
import tempfile
import wave

import numpy as np
import pytest

from analyze_mp3 import (
    analyze_frame,
    analyze_mp3,
    classify_instrument,
    detect_note_durations,
    find_dominant_frequencies,
    freq_to_note,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sine(freq_hz: float, duration: float, sample_rate: int = 22050) -> np.ndarray:
    """Return a mono float32 sine-wave array."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return (np.sin(2 * math.pi * freq_hz * t)).astype(np.float32)


def _write_wav(signal: np.ndarray, sample_rate: int, path: str) -> None:
    """Write a mono float signal as a 16-bit PCM WAV file."""
    pcm = (signal * 32767).astype(np.int16)
    with wave.open(path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


# ---------------------------------------------------------------------------
# freq_to_note
# ---------------------------------------------------------------------------

class TestFreqToNote:
    def test_a4_is_440hz(self):
        note, octave, cents = freq_to_note(440.0)
        assert note == 'A'
        assert octave == 4
        assert abs(cents) < 0.01

    def test_c4_middle_c(self):
        # Middle C ≈ 261.626 Hz
        note, octave, cents = freq_to_note(261.626)
        assert note == 'C'
        assert octave == 4
        assert abs(cents) < 1.0

    def test_a5(self):
        note, octave, _ = freq_to_note(880.0)
        assert note == 'A'
        assert octave == 5

    def test_zero_frequency(self):
        note, octave, cents = freq_to_note(0.0)
        assert note == 'N/A'

    def test_negative_frequency(self):
        note, octave, cents = freq_to_note(-100.0)
        assert note == 'N/A'

    def test_cents_deviation_sharp(self):
        # A4 + 25 cents (clearly still rounds to A4, so cents deviation should be ~+25)
        freq_sharp = 440.0 * (2 ** (25 / 1200))
        _, _, cents = freq_to_note(freq_sharp)
        assert abs(cents - 25.0) < 1.0

    def test_cents_deviation_flat(self):
        # A4 - 25 cents (clearly still rounds to A4, so cents deviation should be ~-25)
        freq_flat = 440.0 * (2 ** (-25 / 1200))
        _, _, cents = freq_to_note(freq_flat)
        assert abs(cents + 25.0) < 1.0

    def test_e4(self):
        # E4 ≈ 329.63 Hz
        note, octave, _ = freq_to_note(329.63)
        assert note == 'E'
        assert octave == 4


# ---------------------------------------------------------------------------
# find_dominant_frequencies
# ---------------------------------------------------------------------------

class TestFindDominantFrequencies:
    SR = 22050

    def test_detects_single_sine(self):
        signal = _make_sine(440.0, 0.5, self.SR)
        peaks = find_dominant_frequencies(signal, self.SR, n_peaks=1)
        assert len(peaks) == 1
        freq, amp = peaks[0]
        assert abs(freq - 440.0) < 5.0, f"Expected ~440 Hz, got {freq:.2f} Hz"
        assert 0 < amp <= 1.0

    def test_detects_two_simultaneous_sines(self):
        """Polyphonic: two sines should produce two detected peaks."""
        s1 = _make_sine(440.0, 0.5, self.SR)
        s2 = _make_sine(880.0, 0.5, self.SR)
        signal = (s1 + s2) * 0.5
        peaks = find_dominant_frequencies(signal, self.SR, n_peaks=2)
        detected_freqs = [p[0] for p in peaks]
        assert any(abs(f - 440.0) < 10 for f in detected_freqs), (
            f"440 Hz not found in {detected_freqs}"
        )
        assert any(abs(f - 880.0) < 10 for f in detected_freqs), (
            f"880 Hz not found in {detected_freqs}"
        )

    def test_silent_frame_returns_empty(self):
        signal = np.zeros(self.SR // 10, dtype=np.float32)
        peaks = find_dominant_frequencies(signal, self.SR)
        assert peaks == []

    def test_empty_frame_returns_empty(self):
        peaks = find_dominant_frequencies(np.array([]), self.SR)
        assert peaks == []

    def test_peaks_sorted_by_amplitude_descending(self):
        # Stronger sine at 440 Hz, weaker at 880 Hz
        s1 = _make_sine(440.0, 0.5, self.SR) * 1.0
        s2 = _make_sine(880.0, 0.5, self.SR) * 0.3
        signal = s1 + s2
        peaks = find_dominant_frequencies(signal, self.SR, n_peaks=2)
        if len(peaks) == 2:
            assert peaks[0][1] >= peaks[1][1], "Peaks should be sorted by amplitude"

    def test_out_of_range_frequency_ignored(self):
        # 15 Hz is below the 20 Hz default min_freq
        signal = _make_sine(15.0, 1.0, self.SR)
        peaks = find_dominant_frequencies(signal, self.SR, min_freq=20.0)
        assert all(p[0] >= 20.0 for p in peaks)


# ---------------------------------------------------------------------------
# classify_instrument
# ---------------------------------------------------------------------------

class TestClassifyInstrument:
    def test_bass_low_frequency_low_centroid(self):
        result = classify_instrument(freq=80.0, spectral_centroid=200.0)
        assert result == 'Bass Guitar'

    def test_flute_high_frequency_high_centroid(self):
        result = classify_instrument(freq=800.0, spectral_centroid=5000.0)
        # Flute centroid range is 2000–9000 Hz; freq 800 Hz is within flute's range too.
        assert result in ('Flute', 'Violin', 'Piano')  # acceptable high-freq instruments

    def test_out_of_range_returns_unknown(self):
        # 10 Hz is outside every instrument's frequency range
        result = classify_instrument(freq=10.0, spectral_centroid=100.0)
        assert result == 'Unknown'

    def test_returns_string(self):
        result = classify_instrument(freq=440.0, spectral_centroid=1000.0)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# detect_note_durations
# ---------------------------------------------------------------------------

class TestDetectNoteDurations:
    def _frame(self, t_start, t_end, notes):
        """Build a minimal frame dict with the given note labels."""
        return {
            'time_start': t_start,
            'time_end': t_end,
            'frequencies': [
                {
                    'note': n,
                    'frequency': 440.0,
                    'amplitude': 0.9,
                    'instrument': 'Piano',
                    'cents_deviation': 0.0,
                }
                for n in notes
            ],
        }

    def test_single_sustained_note(self):
        frames = [
            self._frame(0.0, 0.1, ['A4']),
            self._frame(0.1, 0.2, ['A4']),
            self._frame(0.2, 0.3, ['A4']),
        ]
        events = detect_note_durations(frames)
        assert len(events) == 1
        e = events[0]
        assert e['note'] == 'A4'
        assert e['start_time'] == 0.0
        assert e['end_time'] == 0.3
        assert abs(e['duration'] - 0.3) < 1e-9

    def test_two_sequential_notes(self):
        frames = [
            self._frame(0.0, 0.1, ['A4']),
            self._frame(0.1, 0.2, ['B4']),
        ]
        events = detect_note_durations(frames)
        notes = {e['note'] for e in events}
        assert 'A4' in notes
        assert 'B4' in notes

    def test_two_simultaneous_notes(self):
        frames = [
            self._frame(0.0, 0.1, ['A4', 'C5']),
            self._frame(0.1, 0.2, ['A4', 'C5']),
        ]
        events = detect_note_durations(frames)
        notes = {e['note'] for e in events}
        assert 'A4' in notes
        assert 'C5' in notes

    def test_empty_frames_returns_empty(self):
        assert detect_note_durations([]) == []

    def test_silent_frame_produces_no_events(self):
        frames = [{'time_start': 0.0, 'time_end': 0.1, 'frequencies': []}]
        events = detect_note_durations(frames)
        assert events == []

    def test_note_onset_offset_correct(self):
        frames = [
            self._frame(0.0, 0.1, []),
            self._frame(0.1, 0.2, ['E4']),
            self._frame(0.2, 0.3, ['E4']),
            self._frame(0.3, 0.4, []),
        ]
        events = detect_note_durations(frames)
        assert len(events) == 1
        e = events[0]
        assert e['note'] == 'E4'
        assert e['start_time'] == 0.1
        assert e['end_time'] == 0.3

    def test_sorted_by_start_time(self):
        frames = [
            self._frame(0.0, 0.1, ['C4']),
            self._frame(0.1, 0.2, ['G4']),
        ]
        events = detect_note_durations(frames)
        start_times = [e['start_time'] for e in events]
        assert start_times == sorted(start_times)


# ---------------------------------------------------------------------------
# analyze_frame
# ---------------------------------------------------------------------------

class TestAnalyzeFrame:
    SR = 22050

    def test_returns_dict_with_required_keys(self):
        signal = _make_sine(440.0, 0.1, self.SR)
        result = analyze_frame(signal, self.SR, 0.0, 0.1, n_peaks=1)
        assert 'time_start' in result
        assert 'time_end' in result
        assert 'frequencies' in result

    def test_silent_frame_has_no_frequencies(self):
        signal = np.zeros(self.SR // 10, dtype=np.float32)
        result = analyze_frame(signal, self.SR, 0.0, 0.1)
        assert result['frequencies'] == []

    def test_detects_440hz_note(self):
        signal = _make_sine(440.0, 0.1, self.SR)
        result = analyze_frame(signal, self.SR, 0.0, 0.1, n_peaks=1)
        assert len(result['frequencies']) >= 1
        top = result['frequencies'][0]
        assert abs(top['frequency'] - 440.0) < 10.0
        assert top['note'] == 'A4'
        assert isinstance(top['instrument'], str)

    def test_polyphonic_two_peaks(self):
        s1 = _make_sine(440.0, 0.1, self.SR)
        s2 = _make_sine(880.0, 0.1, self.SR)
        signal = (s1 + s2) * 0.5
        result = analyze_frame(signal, self.SR, 0.0, 0.1, n_peaks=2)
        freqs = [f['frequency'] for f in result['frequencies']]
        assert any(abs(f - 440.0) < 15 for f in freqs)
        assert any(abs(f - 880.0) < 15 for f in freqs)

    def test_time_fields_preserved(self):
        signal = _make_sine(440.0, 0.1, self.SR)
        result = analyze_frame(signal, self.SR, 1.5, 1.6, n_peaks=1)
        assert result['time_start'] == pytest.approx(1.5, abs=1e-3)
        assert result['time_end'] == pytest.approx(1.6, abs=1e-3)


# ---------------------------------------------------------------------------
# analyze_mp3 (end-to-end with a synthetic WAV file)
# ---------------------------------------------------------------------------

class TestAnalyzeMp3:
    SR = 22050

    def _make_wav(self, signal: np.ndarray) -> str:
        fd, path = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
        _write_wav(signal, self.SR, path)
        return path

    def test_basic_structure(self):
        signal = _make_sine(440.0, 0.5, self.SR)
        path = self._make_wav(signal)
        try:
            results = analyze_mp3(path, frame_duration=0.1, n_peaks=2)
        finally:
            os.unlink(path)

        assert results['sample_rate'] == self.SR
        assert abs(results['total_duration'] - 0.5) < 0.02
        assert results['frame_duration'] == 0.1
        assert isinstance(results['frames'], list)
        assert isinstance(results['note_events'], list)

    def test_produces_five_frames_for_half_second(self):
        signal = _make_sine(440.0, 0.5, self.SR)
        path = self._make_wav(signal)
        try:
            results = analyze_mp3(path, frame_duration=0.1, n_peaks=1)
        finally:
            os.unlink(path)

        assert len(results['frames']) == 5

    def test_note_a4_detected(self):
        signal = _make_sine(440.0, 0.5, self.SR)
        path = self._make_wav(signal)
        try:
            results = analyze_mp3(path, frame_duration=0.1, n_peaks=1)
        finally:
            os.unlink(path)

        all_notes = [
            f['note']
            for frame in results['frames']
            for f in frame['frequencies']
        ]
        assert any(n == 'A4' for n in all_notes)

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            analyze_mp3('/nonexistent/path/audio.mp3')

    def test_polyphonic_detection(self):
        """Two simultaneous tones should both appear in the results."""
        s1 = _make_sine(440.0, 0.5, self.SR)
        s2 = _make_sine(880.0, 0.5, self.SR)
        signal = (s1 + s2) * 0.5
        path = self._make_wav(signal)
        try:
            results = analyze_mp3(path, frame_duration=0.1, n_peaks=3)
        finally:
            os.unlink(path)

        all_freqs = [
            f['frequency']
            for frame in results['frames']
            for f in frame['frequencies']
        ]
        assert any(abs(f - 440.0) < 15 for f in all_freqs), (
            f"440 Hz not detected; found: {sorted(set(round(f) for f in all_freqs))}"
        )
        assert any(abs(f - 880.0) < 15 for f in all_freqs), (
            f"880 Hz not detected; found: {sorted(set(round(f) for f in all_freqs))}"
        )

    def test_note_event_duration_approx_signal_length(self):
        signal = _make_sine(440.0, 0.5, self.SR)
        path = self._make_wav(signal)
        try:
            results = analyze_mp3(path, frame_duration=0.1, n_peaks=1)
        finally:
            os.unlink(path)

        durations = [e['duration'] for e in results['note_events'] if e['note'] == 'A4']
        assert durations, "Expected at least one A4 note event"
        # The detected A4 note should span most of the 0.5-second file.
        assert max(durations) > 0.3
