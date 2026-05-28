import numpy as np

from bit_phase.bit_phase import bits_to_phase_wave, bytes_to_bits
from psk.ecc import (
    ECC_SCHEME_HAMMING_7_4,
    MAX_PAYLOAD_LENGTH_BYTES,
    build_header_bits,
    encode_hamming_7_4,
    pad_bits,
)

# Header contains version, ECC scheme, and payload length (bytes).


def encode_dbpsk_list(bits: list[int], previous_bit: int = 1) -> list[int]:
    res = []
    res.append(previous_bit)
    prev: int = previous_bit
    for bit in bits:
        xored = prev ^ bit
        res.append(xored)
        prev = xored
    return res


def encode_bpsk(data: bytes, preamble: list[int]) -> list[int]:
    payload_length_bytes = len(data)
    if payload_length_bytes > MAX_PAYLOAD_LENGTH_BYTES:
        raise ValueError("payload length exceeds header limits.")
    header_bits = build_header_bits(payload_length_bytes, ECC_SCHEME_HAMMING_7_4)
    header_bits = encode_hamming_7_4(header_bits)
    payload_bits, _ = pad_bits(bytes_to_bits(data), 4)
    payload_bits = encode_hamming_7_4(payload_bits)
    bits: list[int] = []
    # Add preamble
    bits += preamble
    bits += header_bits
    bits += payload_bits

    return bits


def encode_to_audio(
    data: bytes,
    preamble: list[int],
    sample_rate: int,
    frequency: int,
    cycles_per_symbol: float,
    algorithm: str,
) -> tuple[np.ndarray, float]:
    if frequency <= 0:
        raise ValueError("frequency must be positive.")
    if cycles_per_symbol <= 0:
        raise ValueError("cycles_per_symbol must be positive.")

    encoded_bits = []
    if algorithm == "bpsk":
        encoded_bits = encode_bpsk(data, preamble)
    elif algorithm == "dbpsk":
        encoded_bits = encode_dbpsk_list(encode_bpsk(data, preamble))
    else:
        raise ValueError(algorithm, "algorithm not recognized")

    waveform = bits_to_phase_wave(
        encoded_bits, frequency, cycles_per_symbol, sample_rate
    )
    duration_seconds = float(waveform.size) / float(sample_rate)
    return waveform, duration_seconds
