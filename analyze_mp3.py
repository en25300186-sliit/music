#!/usr/bin/env python3
"""MP3 Audio Analyzer

Analyzes MP3 files to extract:
- Dominant frequencies at 0.1-second intervals
- Musical notes (with octave and cents deviation)
- Instrument classification using spectral features
- Polyphonic detection (multiple simultaneous notes/instruments)
- Note events with start time, end time, and duration
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import librosa
from scipy.signal import find_peaks


# --- Musical constants ---

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Each instrument profile defines its fundamental frequency range and the
# expected spectral centroid range (both in Hz). These values are used to
# score how well an observed frame matches each instrument category.
INSTRUMENT_PROFILES = {
    'Bass Guitar': {
        'freq_range': (40, 400),
        'centroid_range': (100, 600),
    },
    'Guitar': {
        'freq_range': (80, 1200),
        'centroid_range': (400, 2500),
    },
    'Piano': {
        'freq_range': (27, 4186),
        'centroid_range': (500, 3500),
    },
    'Violin': {
        'freq_range': (196, 3136),
        'centroid_range': (1000, 5500),
    },
    'Flute': {
        'freq_range': (262, 4699),
        'centroid_range': (2000, 9000),
    },
    'Drums/Percussion': {
        'freq_range': (50, 8000),
        'centroid_range': (1500, 7000),
    },
    'Vocals': {
        'freq_range': (80, 1200),
        'centroid_range': (300, 3500),
    },
}


# --- Utility functions ---

def freq_to_note(frequency: float) -> Tuple[str, int, float]:
    """Convert a frequency (Hz) to a musical note name, octave, and cents deviation.

    Returns:
        Tuple of (note_name, octave, cents_deviation).
        cents_deviation is the pitch offset from the nearest semitone in cents
        (100 cents = 1 semitone).  Returns ('N/A', 0, 0.0) for non-positive
        frequencies.
    """
    if frequency <= 0:
        return 'N/A', 0, 0.0

    # A4 = 440 Hz corresponds to MIDI note 69.
    midi_note = 69.0 + 12.0 * np.log2(frequency / 440.0)
    midi_rounded = int(round(midi_note))

    note_name = NOTES[midi_rounded % 12]
    octave = (midi_rounded // 12) - 1
    cents = (midi_note - midi_rounded) * 100.0

    return note_name, octave, float(cents)


def find_dominant_frequencies(
    frame: np.ndarray,
    sample_rate: int,
    n_peaks: int = 5,
    min_freq: float = 20.0,
    max_freq: float = 8000.0,
    min_amplitude_ratio: float = 0.05,
) -> List[Tuple[float, float]]:
    """Find up to *n_peaks* dominant frequencies in an audio frame via FFT.

    A Hann window is applied before computing the FFT to reduce spectral
    leakage.  Peaks are found in the normalised magnitude spectrum and
    filtered by *min_amplitude_ratio* (relative to the strongest component).

    Returns:
        List of (frequency_hz, relative_amplitude) pairs sorted by amplitude
        descending.  The list may be shorter than *n_peaks* if fewer peaks
        are found.
    """
    n = len(frame)
    if n == 0:
        return []

    # Apply Hann window to reduce spectral leakage.
    windowed = frame * np.hanning(n)

    fft_magnitude = np.abs(np.fft.rfft(windowed))
    frequencies = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    # Restrict to the requested frequency band.
    freq_mask = (frequencies >= min_freq) & (frequencies <= max_freq)
    fft_band = fft_magnitude[freq_mask]
    freqs_band = frequencies[freq_mask]

    if len(fft_band) == 0 or np.max(fft_band) == 0:
        return []

    fft_norm = fft_band / np.max(fft_band)

    # Require peaks to be separated by at least 1/200th of the band length so
    # that harmonics close together are not all selected as separate peaks.
    min_distance = max(1, len(fft_norm) // 200)
    peaks, _ = find_peaks(
        fft_norm,
        height=min_amplitude_ratio,
        distance=min_distance,
        prominence=0.02,
    )

    if len(peaks) == 0:
        return []

    # Select the top-N peaks by amplitude.
    amplitudes = fft_norm[peaks]
    top_indices = np.argsort(amplitudes)[-n_peaks:][::-1]

    return [
        (float(freqs_band[peaks[i]]), float(amplitudes[i]))
        for i in top_indices
    ]


def classify_instrument(
    freq: float,
    spectral_centroid: float,
) -> str:
    """Estimate the instrument from a fundamental frequency and spectral centroid.

    Each candidate instrument receives a score based on:
    - Whether *freq* falls within the instrument's typical frequency range.
    - How closely *spectral_centroid* matches the instrument's centroid range.

    Returns the name of the best-matching instrument, or ``'Unknown'``.
    """
    best_name = 'Unknown'
    best_score = -1.0

    for name, profile in INSTRUMENT_PROFILES.items():
        f_min, f_max = profile['freq_range']
        c_min, c_max = profile['centroid_range']

        if not (f_min <= freq <= f_max):
            continue

        # Score based on centroid proximity to the instrument's centroid range.
        if c_min <= spectral_centroid <= c_max:
            score = 1.0
        elif spectral_centroid < c_min:
            score = max(0.0, 1.0 - (c_min - spectral_centroid) / max(c_min, 1))
        else:
            score = max(0.0, 1.0 - (spectral_centroid - c_max) / max(c_max, 1))

        if score > best_score:
            best_score = score
            best_name = name

    return best_name


# --- Frame analysis ---

def analyze_frame(
    frame: np.ndarray,
    sample_rate: int,
    frame_start: float,
    frame_end: float,
    n_peaks: int = 3,
) -> Dict:
    """Analyze a single audio frame and return a structured result dict.

    The result contains the time window and a list of detected frequencies,
    each annotated with its note name, cents deviation, amplitude, and the
    most likely instrument.
    """
    result: Dict = {
        'time_start': round(frame_start, 3),
        'time_end': round(frame_end, 3),
        'frequencies': [],
    }

    # Skip silent frames.
    if len(frame) == 0 or np.max(np.abs(frame)) < 1e-6:
        return result

    freq_amplitudes = find_dominant_frequencies(frame, sample_rate, n_peaks=n_peaks)
    if not freq_amplitudes:
        return result

    # Compute spectral centroid once per frame (used for all peaks).
    centroid_frames = librosa.feature.spectral_centroid(y=frame, sr=sample_rate)
    spectral_centroid = float(np.mean(centroid_frames))

    for freq, amplitude in freq_amplitudes:
        note_name, octave, cents = freq_to_note(freq)
        instrument = classify_instrument(freq, spectral_centroid)

        result['frequencies'].append({
            'frequency': round(freq, 2),
            'amplitude': round(amplitude, 4),
            'note': f"{note_name}{octave}",
            'cents_deviation': round(cents, 1),
            'instrument': instrument,
        })

    return result


# --- Note duration tracking ---

def detect_note_durations(frames_data: List[Dict]) -> List[Dict]:
    """Group consecutive frames containing the same note into note events.

    A note event is created each time a note appears in a frame that was not
    present in the previous frame (onset) and is closed when the note
    disappears (offset).  The duration is ``end_time - start_time``.

    Returns:
        List of note-event dicts sorted by start time.
    """
    note_events: List[Dict] = []
    active: Dict[str, Dict] = {}  # note label -> event dict

    for frame in frames_data:
        current_notes = {info['note'] for info in frame['frequencies']}

        # Open new events for notes that just appeared.
        for info in frame['frequencies']:
            note = info['note']
            if note not in active:
                active[note] = {
                    'note': note,
                    'start_time': frame['time_start'],
                    'frequency': info['frequency'],
                    'amplitude': info['amplitude'],
                    'instrument': info['instrument'],
                    'cents_deviation': info['cents_deviation'],
                }

        # Close events for notes that are no longer present.
        ended = set(active.keys()) - current_notes
        for note in ended:
            event = active.pop(note)
            event['end_time'] = frame['time_start']
            event['duration'] = round(event['end_time'] - event['start_time'], 3)
            note_events.append(event)

    # Close any notes still active at the end of the file.
    if frames_data:
        last_time = frames_data[-1]['time_end']
    else:
        last_time = 0.0

    for event in active.values():
        event['end_time'] = last_time
        event['duration'] = round(last_time - event['start_time'], 3)
        note_events.append(event)

    note_events.sort(key=lambda e: (e['start_time'], e['note']))
    return note_events


# --- Main analysis entry point ---

def analyze_mp3(
    file_path: str,
    frame_duration: float = 0.1,
    n_peaks: int = 3,
) -> Dict:
    """Analyze an MP3 (or any librosa-supported audio) file.

    Args:
        file_path:      Path to the audio file.
        frame_duration: Length of each analysis window in seconds (default 0.1).
        n_peaks:        Maximum number of simultaneous frequencies detected per
                        frame, enabling polyphonic analysis (default 3).

    Returns:
        A dict with keys:
        - ``file``           – input path
        - ``sample_rate``    – original sample rate in Hz
        - ``total_duration`` – file length in seconds
        - ``frame_duration`` – analysis window size in seconds
        - ``frames``         – list of per-frame analysis dicts
        - ``note_events``    – list of note-event dicts with duration info
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"Loading audio file: {file_path}", file=sys.stderr)
    y, sr = librosa.load(file_path, sr=None, mono=True)

    total_duration = len(y) / sr
    print(f"Sample rate      : {sr} Hz", file=sys.stderr)
    print(f"Total duration   : {total_duration:.3f} seconds", file=sys.stderr)
    print(f"Frame size       : {frame_duration} seconds", file=sys.stderr)
    print(f"Max peaks/frame  : {n_peaks}", file=sys.stderr)
    print(file=sys.stderr)

    frame_length = int(frame_duration * sr)
    n_frames = max(1, int(np.ceil(len(y) / frame_length)))

    frames_data: List[Dict] = []
    print(f"Analyzing {n_frames} frames …", file=sys.stderr)

    for i in range(n_frames):
        start_sample = i * frame_length
        end_sample = min(start_sample + frame_length, len(y))
        frame = y[start_sample:end_sample]

        frame_start = i * frame_duration
        frame_end = min((i + 1) * frame_duration, total_duration)

        frames_data.append(
            analyze_frame(frame, sr, frame_start, frame_end, n_peaks=n_peaks)
        )

    note_events = detect_note_durations(frames_data)

    return {
        'file': file_path,
        'sample_rate': int(sr),
        'total_duration': round(total_duration, 3),
        'frame_duration': frame_duration,
        'frames': frames_data,
        'note_events': note_events,
    }


# --- Output formatting ---

def print_results(results: Dict) -> None:
    """Print analysis results in a human-readable table format."""
    sep = '=' * 80

    print(sep)
    print('MP3 AUDIO ANALYSIS RESULTS')
    print(sep)
    print(f"File          : {results['file']}")
    print(f"Sample rate   : {results['sample_rate']} Hz")
    print(f"Total duration: {results['total_duration']} s")
    print(f"Frame size    : {results['frame_duration']} s")
    print()

    print(sep)
    print('FRAME-BY-FRAME FREQUENCY ANALYSIS')
    print(sep)

    for frame in results['frames']:
        if not frame['frequencies']:
            continue

        print(f"\n[{frame['time_start']:.3f}s – {frame['time_end']:.3f}s]")
        for info in frame['frequencies']:
            cents_str = (
                f"  ({info['cents_deviation']:+.1f} ¢)"
                if abs(info['cents_deviation']) > 1
                else ''
            )
            print(
                f"  {info['note']:<5}  {info['frequency']:>8.2f} Hz"
                f"  amp={info['amplitude']:.3f}"
                f"  {info['instrument']}"
                f"{cents_str}"
            )

    print()
    print(sep)
    print('DETECTED NOTE EVENTS  (onset / offset / duration)')
    print(sep)

    if not results['note_events']:
        print('  (no note events detected)')
    else:
        header = (
            f"  {'Note':<6}  {'Freq (Hz)':>10}  "
            f"{'Start (s)':>10}  {'End (s)':>8}  {'Duration (s)':>12}  Instrument"
        )
        print(header)
        print('  ' + '-' * (len(header) - 2))
        for event in results['note_events']:
            print(
                f"  {event['note']:<6}  {event['frequency']:>10.2f}  "
                f"{event['start_time']:>10.3f}  {event['end_time']:>8.3f}  "
                f"{event['duration']:>12.3f}  {event['instrument']}"
            )

    print()


# --- CLI ---

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Analyze MP3 files for frequencies, notes, instruments, and timing.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python analyze_mp3.py song.mp3
  python analyze_mp3.py song.mp3 --frame-duration 0.05 --peaks 5
  python analyze_mp3.py song.mp3 --output json > results.json
""",
    )
    parser.add_argument('file', help='Path to the audio file to analyze (MP3, WAV, FLAC, …)')
    parser.add_argument(
        '--frame-duration',
        type=float,
        default=0.1,
        metavar='SECS',
        help='Analysis window length in seconds (default: 0.1)',
    )
    parser.add_argument(
        '--peaks',
        type=int,
        default=3,
        metavar='N',
        help='Max simultaneous frequencies per frame – enables polyphonic analysis (default: 3)',
    )
    parser.add_argument(
        '--output',
        choices=['text', 'json'],
        default='text',
        help='Output format (default: text)',
    )

    args = parser.parse_args()

    try:
        results = analyze_mp3(
            args.file,
            frame_duration=args.frame_duration,
            n_peaks=args.peaks,
        )

        if args.output == 'json':
            print(json.dumps(results, indent=2))
        else:
            print_results(results)

    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except (IOError, OSError, ValueError, RuntimeError) as exc:
        print(f"Error analyzing file: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
