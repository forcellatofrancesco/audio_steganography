import numpy as np

from bit_phase.bit_phase import bits_to_phase_wave, bytes_to_bits, int_to_bit_list

# TODO: it should be added an header function:
# - version
# - algorithm
# - payload length


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
    # 16 bits representing the length of the payload (like IPv4)
    payload_length = int_to_bit_list(len(data))
    bits: list[int] = []
    # Add preamble
    bits += preamble
    # Convert bytes to list of bits
    bits += payload_length
    bits += bytes_to_bits(data)

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
