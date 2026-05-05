import numpy as np
from scipy.signal import butter, hilbert, sosfilt, sosfiltfilt

from bit_phase.bit_phase import bit_to_phase_wave, bits_to_bytes
from psk.psk_encoder import encode_dbpsk_list


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
    # non serve
    sos = butter(
        filter_order, [low, high], btype="bandpass", fs=sample_rate, output="sos"
    )
    waveform_f64 = waveform.astype(np.float64)
    try:
        filtered = sosfiltfilt(sos, waveform_f64)
    except ValueError:
        filtered = sosfilt(sos, waveform_f64)
    return np.asarray(hilbert(filtered), dtype=np.complex128)


def detect_preamble(
    waveform: np.ndarray,
    preamble: list[int],
    sample_rate=44100,
    frequency=440,
    cycles_per_symbol=1.0,
    algorithm: str = "dbpsk",
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
    if algorithm == "bpsk":
        encoded_preamble = preamble
    elif algorithm == "dbpsk":
        encoded_preamble = encode_dbpsk_list(preamble)
    else:
        raise ValueError(algorithm, "algorithm not recognized")

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
    # TODO: filtered_waveform should be useless
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
    filtered_waveform = waveform
    expected = expected - np.mean(expected)

    # TODO: check if the division by correlation works
    candidate = filtered_waveform - np.mean(filtered_waveform)
    candidate = candidate / np.var(candidate)
    correlation = np.correlate(candidate, expected, mode="valid")
    correlation = correlation / np.var(correlation)

    if correlation.size == 0:
        return 0

    return int(np.argmax(np.abs(correlation)))


def decode_from_audio(
    waveform,
    preamble: list[int],
    sample_rate: int,
    frequency: int,
    cycles_per_symbol: float,
    algorithm: str,
):
    if frequency <= 0:
        raise ValueError("frequency must be positive.")
    if cycles_per_symbol <= 0:
        raise ValueError("cycles_per_symbol must be positive.")
    if algorithm not in {"bpsk", "dbpsk"}:
        raise ValueError(algorithm, "algorithm not recognized")

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
        algorithm=algorithm,
    )
    trimmed_waveform = analytic_waveform[start_index:]

    samples_per_symbol = max(1, int(round(sample_rate * cycles_per_symbol / frequency)))
    if trimmed_waveform.size < samples_per_symbol:
        return b""

    symbol_values = []
    for start in range(
        0, trimmed_waveform.size - samples_per_symbol + 1, samples_per_symbol
    ):
        segment = trimmed_waveform[start : start + samples_per_symbol]
        if segment.size < samples_per_symbol:
            break
        t = (np.arange(samples_per_symbol) + start) / sample_rate
        baseband = segment * np.exp(-1j * 2 * np.pi * frequency * t)
        symbol_values.append(np.mean(baseband))

    if len(symbol_values) < 2:
        return b""

    decoded_bits = []
    # TODO: refactor this function
    if algorithm == "dbpsk":
        previous_phase = np.angle(symbol_values[0])
        for symbol_value in symbol_values[1:]:
            phase = np.angle(symbol_value)
            phase_delta = np.angle(np.exp(1j * (phase - previous_phase)))
            decoded_bits.append(0 if np.cos(phase_delta) >= 0 else 1)
            previous_phase = phase
    elif algorithm == "bpsk":
        if len(symbol_values) < len(preamble):
            return b""

        eps = 1e-12
        normalized_symbols = [value / (np.abs(value) + eps) for value in symbol_values]

        preamble_symbols = normalized_symbols[: len(preamble)]
        ref0_symbols = [
            symbol for bit, symbol in zip(preamble, preamble_symbols) if bit == 0
        ]
        ref1_symbols = [
            symbol for bit, symbol in zip(preamble, preamble_symbols) if bit == 1
        ]

        ref0 = np.mean(ref0_symbols) if ref0_symbols else None
        ref1 = np.mean(ref1_symbols) if ref1_symbols else None

        if ref0 is not None:
            ref0 = ref0 / (np.abs(ref0) + eps)
        if ref1 is not None:
            ref1 = ref1 / (np.abs(ref1) + eps)

        for symbol in normalized_symbols:
            if ref0 is not None and ref1 is not None:
                score0 = np.real(symbol * np.conj(ref0))
                score1 = np.real(symbol * np.conj(ref1))
                decoded_bits.append(0 if score0 >= score1 else 1)
            elif ref0 is not None:
                score0 = np.real(symbol * np.conj(ref0))
                decoded_bits.append(0 if score0 >= 0 else 1)
            elif ref1 is not None:
                score1 = np.real(symbol * np.conj(ref1))
                decoded_bits.append(1 if score1 >= 0 else 0)
            else:
                decoded_bits.append(0)

    # Strip leading/trailing preamble
    payload_bits = decoded_bits[len(preamble) :]
    if (
        len(payload_bits) >= len(preamble)
        and payload_bits[-len(preamble) :] == preamble
    ):
        payload_bits = payload_bits[: -len(preamble)]

    return bits_to_bytes(payload_bits)
