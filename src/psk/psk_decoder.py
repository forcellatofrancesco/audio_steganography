import numpy as np

from bit_phase.bit_phase import bit_to_phase_wave, bits_to_bytes
from psk.ecc import (
    ECC_SCHEME_HAMMING_7_4,
    ECC_SCHEME_NONE,
    ENCODED_HEADER_BITS,
    HEADER_DATA_BITS,
    HEADER_VERSION,
    decode_hamming_7_4,
    parse_header_bits,
)
from psk.psk_encoder import encode_dbpsk_list


def decode_symbol_values(
    symbol_values: list[complex],
    preamble: list[int],
    algorithm: str,
) -> list[int]:
    decoded_bits = []
    if algorithm == "dbpsk":
        previous_phase = np.angle(symbol_values[0])
        for symbol_value in symbol_values[1:]:
            phase = np.angle(symbol_value)
            phase_delta = np.angle(np.exp(1j * (phase - previous_phase)))
            decoded_bits.append(0 if np.cos(phase_delta) >= 0 else 1)
            previous_phase = phase
    elif algorithm == "bpsk":
        eps = 1e-12
        if len(symbol_values) == 0:
            return []

        normalized_symbols = np.asarray(
            [value / (np.abs(value) + eps) for value in symbol_values],
            dtype=np.complex128,
        )
        preamble_len = min(len(preamble), normalized_symbols.size)
        preamble_symbols = normalized_symbols[:preamble_len]
        if preamble_len:
            bit_signs = np.array(
                [1.0 if bit == 0 else -1.0 for bit in preamble[:preamble_len]],
                dtype=np.float64,
            )
            ref = np.sum(preamble_symbols * bit_signs)
        else:
            ref = 0.0

        if np.abs(ref) >= eps:
            ref = ref / (np.abs(ref) + eps)
            for symbol in normalized_symbols:
                score = np.real(symbol * np.conj(ref))
                decoded_bits.append(0 if score >= 0 else 1)
        else:
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
    else:
        raise ValueError(algorithm, "algorithm not recognized")

    return decoded_bits


def detect_preamble(
    waveform: np.ndarray,
    preamble: list[int],
    sample_rate=44100,
    frequency=440,
    cycles_per_symbol=1.0,
    algorithm: str = "dbpsk",
) -> tuple[int, bool]:
    """
    Find the sample offset of a DBPSK preamble using matched filtering.

    Args:
            waveform (np.ndarray): Input audio samples.
            preamble (list[int]): Preamble bits before differential encoding.
            sample_rate (int): Sample rate in Hz.
            frequency (float): Carrier frequency in Hz.
            cycles_per_symbol (float): Carrier cycles per symbol.

    Returns:
            tuple[int, bool]: Sample index of the best preamble match and a boolean
                             indicating whether the preamble was detected.
    """
    if waveform.size == 0:
        return 0, False

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
        return 0, False

    # Direct matched filtering on the raw waveform.
    # This is equivalent to sliding a dot product over the signal, but faster and
    # without the bandpass/Hilbert preprocessing.
    candidate = np.real(np.asarray(waveform))
    candidate = candidate - np.mean(candidate)

    expected = expected - np.mean(expected)
    expected_analytic = np.real(np.asarray(expected))

    win_len = expected_analytic.size
    if candidate.size < win_len:
        return 0, False

    # Fast sliding matched filter.
    raw_scores = np.correlate(candidate, expected_analytic, mode="valid")
    template_energy = np.sum(np.abs(expected_analytic) ** 2)
    if template_energy == 0:
        return 0, False

    candidate_power = np.abs(candidate) ** 2
    candidate_energy = np.convolve(candidate_power, np.ones(win_len), mode="valid")
    denom = np.sqrt(candidate_energy * template_energy)
    scores = np.divide(
        raw_scores,
        denom,
        out=np.zeros_like(raw_scores, dtype=np.float64),
        where=denom > 0,
    )

    # pick the offset with maximum absolute normalized inner product
    best = int(np.argmax(np.abs(scores)))

    # Deterministically verify the preamble by decoding it bit-by-bit
    trimmed_waveform = np.asarray(waveform[best:], dtype=np.float64)

    if trimmed_waveform.size < samples_per_symbol * len(encoded_preamble):
        return best, False

    # Extract symbols from the preamble region
    symbol_values = []
    for i in range(len(encoded_preamble)):
        start = i * samples_per_symbol
        end = start + samples_per_symbol
        segment = trimmed_waveform[start:end]
        if segment.size < samples_per_symbol:
            return best, False
        t = np.arange(samples_per_symbol) / sample_rate
        mixed = segment * np.exp(-1j * 2 * np.pi * frequency * t)
        symbol_values.append(np.sum(mixed) / samples_per_symbol)

    # Compare decoded preamble with the original preamble
    decoded_bits = decode_symbol_values(symbol_values, preamble, algorithm)
    detected = decoded_bits == preamble

    return best, detected


def decode_from_audio(
    waveform,
    preamble: list[int],
    sample_rate: int,
    frequency: int,
    cycles_per_symbol: float,
    algorithm: str,
) -> tuple[bytes, str]:
    if frequency <= 0:
        raise ValueError("frequency must be positive.")
    if cycles_per_symbol <= 0:
        raise ValueError("cycles_per_symbol must be positive.")
    if algorithm not in {"bpsk", "dbpsk"}:
        raise ValueError(algorithm, "algorithm not recognized")

    start_index, preamble_detected = detect_preamble(
        waveform,
        preamble,
        sample_rate=sample_rate,
        frequency=frequency,
        cycles_per_symbol=cycles_per_symbol,
        algorithm=algorithm,
    )

    # Return empty bytes if preamble was not detected
    if not preamble_detected:
        return b"", "not-valid-preamble"

    trimmed_waveform = np.asarray(waveform[start_index:], dtype=np.float64)

    samples_per_symbol = max(1, int(round(sample_rate * cycles_per_symbol / frequency)))
    if trimmed_waveform.size < samples_per_symbol:
        return b"", "check-error"

    symbol_values = []
    for start in range(
        0, trimmed_waveform.size - samples_per_symbol + 1, samples_per_symbol
    ):
        segment = trimmed_waveform[start : start + samples_per_symbol]
        if segment.size < samples_per_symbol:
            break
        t = (np.arange(samples_per_symbol) + start) / sample_rate
        mixed = segment * np.exp(-1j * 2 * np.pi * frequency * t)
        symbol_values.append(np.sum(mixed) / samples_per_symbol)

    if len(symbol_values) < 2:
        return b"", "check-error"

    if algorithm == "bpsk" and len(symbol_values) < len(preamble):
        return b"", "check-error"
    decoded_bits = decode_symbol_values(symbol_values, preamble, algorithm)
    # Strip leading preamble
    payload_bits = decoded_bits[len(preamble) :]
    if len(payload_bits) < ENCODED_HEADER_BITS:
        return b"", "no-header-in-payload"

    header_bits, _ = decode_hamming_7_4(payload_bits[:ENCODED_HEADER_BITS])
    if len(header_bits) < HEADER_DATA_BITS:
        return b"", "no-header-in-payload"

    version, ecc_scheme, payload_length_bytes = parse_header_bits(header_bits)
    if version != HEADER_VERSION:
        return b"", "unsupported-header-version"
    if ecc_scheme not in {ECC_SCHEME_NONE, ECC_SCHEME_HAMMING_7_4}:
        return b"", "unsupported-ecc-scheme"

    payload_bits = payload_bits[ENCODED_HEADER_BITS:]
    if ecc_scheme == ECC_SCHEME_NONE:
        decoded_payload_bits = payload_bits
    else:
        decoded_payload_bits, _ = decode_hamming_7_4(payload_bits)

    payload_length_bits = payload_length_bytes * 8
    if len(decoded_payload_bits) < payload_length_bits:
        return bits_to_bytes(decoded_payload_bits), "paylod-too-short"

    decoded_payload_bits = decoded_payload_bits[:payload_length_bits]

    return bits_to_bytes(decoded_payload_bits), "valid-preamble"
