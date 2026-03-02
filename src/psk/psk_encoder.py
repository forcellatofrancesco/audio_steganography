import numpy as np
from psk.utils import bit_to_phase_wave, bits_to_phase_wave, bytes_to_bits


def encode_dbpsk_list(bits: list[int], previous_bit: int = 1) -> list[int]:
    res = []
    res.append(previous_bit)
    prev: int = previous_bit
    for bit in bits:
        xored = prev ^ bit
        res.append(xored)
        prev = xored
    return res


def encode_bpsk(data: bytes, preamble: list[int], previous_bit: int = 1) -> list[int]:
    bits: list[int] = []
    # Add preamble
    bits += preamble
    # Convert bytes to list of bits
    bits += bytes_to_bits(data)
    # Add preamble as ending sequence
    bits += preamble
    return bits


def encode_to_audio(
    data: bytes,
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

    encoded_bits = []
    if algorithm == "bpsk":
        encoded_bits = encode_bpsk(data, preamble)
    elif algorithm == "dbpsk":
        encoded_bits = encode_dbpsk_list(encode_bpsk(data, preamble))
    else:
        raise ValueError(algorithm, "algorithm not recognized")
    return bits_to_phase_wave(encoded_bits, frequency, cycles_per_symbol, sample_rate)
