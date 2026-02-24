import numpy as np
from scipy.signal import butter, hilbert, sosfilt, sosfiltfilt

from psk.utils import bits_to_bytes

from .psk_encoder import (
    bit_to_phase_wave,
    encode_dbpsk_list,
)


def _compute_band_edges(frequency, sample_rate, cycles_per_symbol):
    nyquist = sample_rate / 2.0
    symbol_rate = frequency / cycles_per_symbol
    bandwidth = max(symbol_rate * 2.0, frequency * 0.2)
    low = max(1.0, frequency - bandwidth / 2.0)
    high = min(nyquist * 0.98, frequency + bandwidth / 2.0)
    if low >= high:
        high = min(nyquist * 0.98, frequency * 1.1)
        low = max(1.0, frequency * 0.9)
    return low, high


def _prepare_analytic_signal(
    waveform,
    sample_rate,
    frequency,
    cycles_per_symbol,
    filter_order=6,
):
    if waveform.size == 0:
        return np.asarray(waveform, dtype=np.complex64)

    low, high = _compute_band_edges(frequency, sample_rate, cycles_per_symbol)
    sos = butter(
        filter_order, [low, high], btype="bandpass", fs=sample_rate, output="sos"
    )
    waveform_f64 = waveform.astype(np.float64)
    try:
        filtered = sosfiltfilt(sos, waveform_f64)
    except ValueError:
        filtered = sosfilt(sos, waveform_f64)
    return np.asarray(hilbert(filtered), dtype=np.complex128)


def _symbol_phase(analytic_waveform, start, samples_per_symbol, sample_rate, frequency):
    segment = analytic_waveform[start : start + samples_per_symbol]
    if segment.size < samples_per_symbol:
        return None

    t = (np.arange(samples_per_symbol) + start) / sample_rate
    baseband = segment * np.exp(-1j * 2 * np.pi * frequency * t)
    return np.angle(np.mean(baseband))


def decode(bits: list[int]) -> list[int]:
    previous_bit = bits[0]
    xored_bits = []
    for bit in bits[1:]:
        xored = bit ^ previous_bit
        previous_bit = bit
        xored_bits.append(xored)
    return xored_bits


def detect_preamble(
    waveform,
    preamble: list[int],
    sample_rate=44100,
    frequency=440,
    cycles_per_symbol=1.0,
) -> int:
    """
    Find the sample offset of a DBPSK preamble using matched filtering.

    Args:
            waveform (np.ndarray): Input audio samples.
            preamble (list[int]): Preamble bits before differential encoding.
            sample_rate (int): Sample rate in Hz.
            frequency (float): Carrier frequency in Hz.
            cycles_per_symbol (float): Carrier cycles per symbol.

    Returns:
            int: Sample index of the best preamble match.
    """
    if waveform.size == 0:
        return 0

    samples_per_symbol = max(1, int(round(sample_rate * cycles_per_symbol / frequency)))
    encoded_preamble = encode_dbpsk_list(preamble)

    expected_chunks = []
    sample_offset = 0
    for bit in encoded_preamble:
        expected_chunks.append(
            bit_to_phase_wave(
                bit,
                frequency,
                samples_per_symbol,
                sample_offset,
                sample_rate,
            )
        )
        sample_offset += samples_per_symbol

    expected = (
        np.concatenate(expected_chunks).astype(np.float32)
        if expected_chunks
        else np.array([], dtype=np.float32)
    )
    if expected.size == 0 or waveform.size < expected.size:
        return 0

    filtered_waveform = np.asarray(
        np.real(
            _prepare_analytic_signal(
                waveform,
                sample_rate,
                frequency,
                cycles_per_symbol,
            )
        ),
        dtype=np.float64,
    )

    expected = expected - np.mean(expected)
    candidate = filtered_waveform - np.mean(filtered_waveform)
    correlation = np.correlate(candidate, expected, mode="valid")
    if correlation.size == 0:
        return 0

    return int(np.argmax(np.abs(correlation)))


def decode_phase_shift_keying(
    waveform,
    preamble: list[int],
    sample_rate=44100,
    frequency=440,
    cycles_per_symbol=1.0,
):
    if frequency <= 0:
        raise ValueError("frequency must be positive.")
    if cycles_per_symbol <= 0:
        raise ValueError("cycles_per_symbol must be positive.")
    analytic_waveform = np.asarray(
        _prepare_analytic_signal(
            waveform,
            sample_rate,
            frequency,
            cycles_per_symbol,
        ),
        dtype=np.complex128,
    )
    start_index = detect_preamble(
        waveform,
        preamble,
        sample_rate=sample_rate,
        frequency=frequency,
        cycles_per_symbol=cycles_per_symbol,
    )
    trimmed_waveform = analytic_waveform[start_index:]

    samples_per_symbol = max(1, int(round(sample_rate * cycles_per_symbol / frequency)))
    if trimmed_waveform.size < samples_per_symbol:
        return b""

    symbol_phases = []
    for start in range(
        0, trimmed_waveform.size - samples_per_symbol + 1, samples_per_symbol
    ):
        phase = _symbol_phase(
            trimmed_waveform, start, samples_per_symbol, sample_rate, frequency
        )
        if phase is None:
            break
        symbol_phases.append(phase)

    if len(symbol_phases) < 2:
        return b""

    decoded_bits = []
    previous_phase = symbol_phases[0]
    for phase in symbol_phases[1:]:
        phase_delta = np.angle(np.exp(1j * (phase - previous_phase)))
        decoded_bits.append(0 if np.cos(phase_delta) >= 0 else 1)
        previous_phase = phase

    # Strip leading/trailing preamble
    payload_bits = decoded_bits[len(preamble) :]
    if (
        len(payload_bits) >= len(preamble)
        and payload_bits[-len(preamble) :] == preamble
    ):
        payload_bits = payload_bits[: -len(preamble)]

    return bits_to_bytes(payload_bits)
