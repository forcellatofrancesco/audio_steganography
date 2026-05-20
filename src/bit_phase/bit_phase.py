import numpy as np


def bit_to_phase_wave(
    bit: int,
    frequency,
    samples_per_symbol,
    sample_offset,
    sample_rate,
):
    phase = np.pi * bit  # phase = 0 when xored = 0, phase = pi when xored = 1
    t_symbol = (np.arange(samples_per_symbol) + sample_offset) / sample_rate
    symbol = np.sin(2 * np.pi * frequency * t_symbol + phase)
    return symbol


def bits_to_phase_wave(
    bits: list[int],
    frequency: int,
    cycles_per_symbol: float,
    sample_rate: int,
) -> np.ndarray:
    waveform_chunks = []
    sample_offset = 0
    samples_per_symbol = max(1, int(round(sample_rate * cycles_per_symbol / frequency)))
    for bit in bits:
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


def bytes_to_bits(data: bytes) -> list[int]:
    bits = []
    for byte in data:
        for bit_index in range(7, -1, -1):
            bit = (byte >> bit_index) & 1
            bits.append(bit)
    return bits


def bits_to_bytes(bits: list[int]) -> bytes:
    byte_values = []
    current = 0
    for idx, bit in enumerate(bits):
        current = (current << 1) | bit
        if (idx + 1) % 8 == 0:
            byte_values.append(current)
            current = 0
    return bytes(byte_values)


def int_to_bit_list(x: int, bits: int = 16) -> list[int]:
    return [(x >> (bits - 1 - i)) & 1 for i in range(bits)]


def bits_to_int(bits: list[int]) -> int:
    return int("".join(map(str, bits)), 2)
