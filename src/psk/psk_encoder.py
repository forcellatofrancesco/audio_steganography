import numpy as np
from psk.utils import bit_to_phase_wave, bytes_to_bits


def encode_dbpsk_list(bits: list[int], previous_bit: int = 1) -> list[int]:
    res = []
    res.append(previous_bit)
    prev: int = previous_bit
    for bit in bits:
        xored = prev ^ bit
        res.append(xored)
        prev = xored
    return res


def encode_dbpsk(data: bytes, preamble: list[int], previous_bit: int = 1) -> list[int]:
    bits: list[int] = []
    # Add preamble
    bits += preamble
    # Convert bytes to list of bits
    bits += bytes_to_bits(data)
    # Add preamble as ending sequence
    bits += preamble
    # Encode in differential
    encoded: list[int] = encode_dbpsk_list(bits, previous_bit)
    return encoded


def differential_binary_phase_shift_keying(
    data: bytes,
    preamble: list[int],
    sample_rate=44100,
    frequency=440,
    cycles_per_symbol=1.0,
):
    if frequency <= 0:
        raise ValueError("frequency must be positive.")
    if cycles_per_symbol <= 0:
        raise ValueError("cycles_per_symbol must be positive.")
    samples_per_symbol = max(1, int(round(sample_rate * cycles_per_symbol / frequency)))
    waveform_chunks = []
    encoded_bits = encode_dbpsk(data, preamble)
    sample_offset = 0
    for bit in encoded_bits:
        symbol = bit_to_phase_wave(
            bit,
            frequency,
            samples_per_symbol,
            sample_offset,
            sample_rate,
        )
        waveform_chunks.append(symbol)
        sample_offset += samples_per_symbol
    return np.concatenate(waveform_chunks) if waveform_chunks else np.array([])
