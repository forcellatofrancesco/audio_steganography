import numpy as np

from audio.audiowaves import low_pass_filter
from bit_phase.bit_phase import bits_to_bytes, bits_to_phase_wave
from psk.ecc import (
    ECC_SCHEME_HAMMING_7_4,
    ECC_SCHEME_NONE,
    ENCODED_HEADER_BITS,
    HEADER_DATA_BITS,
    HEADER_VERSION,
    decode_hamming_7_4,
    parse_header_bits,
)
from psk.psk_encoder import encode_dbpsk_list, encode_to_audio
from util.types import bit_array, value_array


def decode_symbol_values_dbpsk(
    trimmed_waveform: value_array,
    samples_per_symbol: int,
    sample_rate: int,
    frequency: int,
) -> value_array:
    waveform = np.asarray(trimmed_waveform, dtype=np.float64)
    usable_len = waveform.size - (waveform.size % samples_per_symbol)
    if usable_len < samples_per_symbol * 2:
        return np.array([], dtype=np.int8)

    time_vector = np.arange(usable_len) / sample_rate
    osc = np.exp(-1j * 2 * np.pi * frequency * time_vector)
    mixed = waveform[:usable_len] * osc
    symbol_values = mixed.reshape(-1, samples_per_symbol).mean(axis=1)

    phase_deltas = np.angle(np.exp(1j * np.diff(np.angle(symbol_values))))
    return np.where(np.cos(phase_deltas) >= 0, -1, 1).astype(np.int8)


def decode_symbol_values_bpsk(
    trimmed_waveform: value_array,
    samples_per_symbol: int,
    sample_rate: int,
    frequency: int,
) -> value_array:
    waveform = np.asarray(trimmed_waveform, dtype=np.float64)
    if waveform.size == 0:
        return np.array([], dtype=np.int8)

    time_vector = np.arange(waveform.size) / sample_rate
    osc = np.exp(-1j * 2 * np.pi * time_vector * frequency)
    baseband = low_pass_filter(osc * waveform, sample_rate)
    phase = np.angle(baseband)
    # TODO: remove
    ###################################################################
    # global out_phase
    # out_phase = phase
    ###################################################################
    start_indices = np.arange(
        0, phase.size - samples_per_symbol + 1, samples_per_symbol
    )
    if start_indices.size == 0:
        return np.array([], dtype=np.int8)

    middle_indices = start_indices + samples_per_symbol // 2
    middle_indices = middle_indices[middle_indices < phase.size]
    if middle_indices.size == 0:
        return np.array([], dtype=np.int8)

    return np.where(phase[middle_indices] < 0, -1, 1).astype(np.int8)


def detect_preamble(
    waveform: value_array,
    preamble: bit_array,
    sample_rate=44100,
    frequency=440,
    cycles_per_symbol=1.0,
    algorithm: str = "dbpsk",
) -> int:
    """
    Find the sample offset of a DBPSK preamble using matched filtering.

    Args:
            waveform (value_array): Input audio samples.
            preamble (bit_array): Preamble bits before differential encoding.
            sample_rate (int): Sample rate in Hz.
            frequency (float): Carrier frequency in Hz.
            cycles_per_symbol (float): Carrier cycles per symbol.

    Returns:
            tuple[int, bool]: Sample index of the best preamble match and a boolean
                             indicating whether the preamble was detected.
    """
    if waveform.size == 0:
        return 0

    preamble = np.asarray(preamble, dtype=np.int8).reshape(-1)
    if algorithm == "bpsk":
        encoded_preamble = preamble
    elif algorithm == "dbpsk":
        encoded_preamble = encode_dbpsk_list(np.asarray(preamble, dtype=np.int8))
    else:
        raise ValueError(algorithm, "algorithm not recognized")

    expected = bits_to_phase_wave(
        encoded_preamble, frequency, cycles_per_symbol, sample_rate
    )

    if expected.size == 0 or waveform.size < expected.size:
        return 0

    # Direct matched filtering on the raw waveform.
    # This is equivalent to sliding a dot product over the signal, but faster and
    # without the bandpass/Hilbert preprocessing.
    candidate = np.real(np.asarray(waveform))
    candidate = candidate - np.mean(candidate)

    expected = expected - np.mean(expected)
    expected_analytic = np.real(np.asarray(expected))
    win_len = expected_analytic.size
    if candidate.size < win_len:
        return 0

    # Fast sliding matched filter.
    raw_scores = np.correlate(candidate, expected_analytic, mode="same")
    template_energy = np.sum(np.abs(expected_analytic) ** 2)
    if template_energy == 0:
        return 0

    candidate_power = np.abs(candidate) ** 2
    candidate_energy = np.convolve(candidate_power, np.ones(win_len), mode="same")
    denom = np.sqrt(candidate_energy * template_energy)
    scores = np.divide(
        raw_scores,
        denom,
        out=np.zeros_like(raw_scores, dtype=np.float64),
        where=denom > 0,
    )

    # pick the offset with maximum absolute normalized inner product

    best = int(np.argmax(np.abs(scores)))
    best = best - expected_analytic.shape[0] // 2
    return best


def convert_to_1_1(a: bit_array) -> value_array:
    return np.where(a > 0, 1, -1).astype(np.int8)


def convert_to_0_1(a: value_array) -> bit_array:
    return np.where(a > 0, 1, 0).astype(np.int8)

# TODO: remove
# out_trimmed_waveform = None
# out_phase = None
# out_expected = None


def decode_from_audio(
    waveform: value_array,
    preamble: bit_array,
    sample_rate: int,
    frequency: int,
    cycles_per_symbol: float,
    algorithm: str,
) -> tuple[bytes, str, int]:
    if frequency <= 0:
        raise ValueError("frequency must be positive.")
    if cycles_per_symbol <= 0:
        raise ValueError("cycles_per_symbol must be positive.")
    if algorithm not in {"bpsk", "dbpsk"}:
        raise ValueError(algorithm, "algorithm not recognized")

    preamble = np.asarray(preamble, dtype=np.int8).reshape(-1)

    start_index = detect_preamble(
        waveform,
        preamble,
        sample_rate=sample_rate,
        frequency=frequency,
        cycles_per_symbol=cycles_per_symbol,
        algorithm=algorithm,
    )

    trimmed_waveform = np.asarray(waveform[start_index:], dtype=np.float64)
    ###################################################################
    # global out_trimmed_waveform
    # out_trimmed_waveform = trimmed_waveform
    ###################################################################
    samples_per_symbol = max(1, int(round(sample_rate * cycles_per_symbol / frequency)))
    decoded_bits = np.array([], dtype=np.int8)
    if algorithm == "dbpsk":
        decoded_bits = decode_symbol_values_dbpsk(
            trimmed_waveform,
            samples_per_symbol,
            sample_rate,
            frequency,
        )
    elif algorithm == "bpsk":
        decoded_bits = decode_symbol_values_bpsk(
            trimmed_waveform,
            samples_per_symbol,
            sample_rate,
            frequency,
        )

    if decoded_bits.shape[0] < len(preamble):
        return b"", "not-valid-preamble", start_index

    ######################################################################################
    # Per il messaggio HI!
    # 1. forma d'onda sengale whatsapp
    # 2. trimmed_waveform
    # 3. sopra il segnale della fase
    # 4. sopra di quello il segnale che mi aspetto di avere (come un'onda quadra)
    ######################################################################################

    preamble_values = convert_to_1_1(np.asarray(preamble))
    checksum = np.sum(preamble_values * decoded_bits[: len(preamble)])
    if (checksum > 0 and checksum < 11) or (checksum < 0 and checksum > -11):
        return b"", "not-valid-preamble", start_index

    if checksum < 0:
        decoded_bits = decoded_bits * -1

    # Strip leading preamble
    payload_bits = convert_to_0_1(decoded_bits[len(preamble) :])
    if len(payload_bits) < ENCODED_HEADER_BITS:
        return b"", "no-header-in-payload", start_index

    header_bits, _ = decode_hamming_7_4(payload_bits[:ENCODED_HEADER_BITS])
    if len(header_bits) < HEADER_DATA_BITS:
        return b"", "no-header-in-payload", start_index

    version, ecc_scheme, payload_length_bytes = parse_header_bits(header_bits)
    if version != HEADER_VERSION:
        return b"", "unsupported-header-version", start_index
    if ecc_scheme not in {ECC_SCHEME_NONE, ECC_SCHEME_HAMMING_7_4}:
        return b"", "unsupported-ecc-scheme", start_index

    payload_bits = payload_bits[ENCODED_HEADER_BITS:]
    if ecc_scheme == ECC_SCHEME_NONE:
        decoded_payload_bits = payload_bits
    else:
        decoded_payload_bits, _ = decode_hamming_7_4(payload_bits)
    decoded_payload_bits = np.asarray(decoded_payload_bits, dtype=np.int8)

    payload_length_bits = payload_length_bytes * 8
    if len(decoded_payload_bits) < payload_length_bits:
        return bits_to_bytes(decoded_payload_bits), "paylod-too-short", start_index

    decoded_payload_bits = decoded_payload_bits[:payload_length_bits]
    # TODO: remove
    ####################################################################
    # global out_expected
    # out_expected, _ = encode_to_audio(
    #     "Hi!".encode("utf-8"),
    #     preamble,
    #     sample_rate,
    #     frequency,
    #     cycles_per_symbol,
    #     algorithm,
    # )
    # message = bits_to_bytes(decoded_payload_bits).decode("utf-8", errors="replace")
    # print(message)
    ###################################################################
    return bits_to_bytes(decoded_payload_bits), "valid-preamble", start_index
