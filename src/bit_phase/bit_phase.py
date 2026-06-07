import numpy as np


def bit_to_phase_wave(
    bit: int,
    frequency,
    samples_per_symbol,
    sample_offset,
    sample_rate,
):
    bit = int(bit)
    phase = np.pi * bit  # phase = 0 when xored = 0, phase = pi when xored = 1
    t_symbol = (np.arange(samples_per_symbol) + sample_offset) / sample_rate
    symbol = np.sin(2 * np.pi * frequency * t_symbol + phase)
    return symbol


def bits_to_phase_wave(
    bits: np.ndarray,
    frequency: int,
    cycles_per_symbol: float,
    sample_rate: int,
) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.int8).reshape(-1)
    samples_per_symbol = max(1, int(round(sample_rate * cycles_per_symbol / frequency)))
    waveform = np.empty(bits.size * samples_per_symbol, dtype=np.float64)
    for idx, bit in enumerate(bits):
        sample_offset = idx * samples_per_symbol
        waveform[sample_offset : sample_offset + samples_per_symbol] = (
            bit_to_phase_wave(
                bit,
                frequency,
                samples_per_symbol,
                sample_offset,
                sample_rate,
            )
        )
    return waveform


def bytes_to_bits(data: bytes) -> np.ndarray:
    byte_values = np.frombuffer(data, dtype=np.uint8)
    if byte_values.size == 0:
        return np.array([], dtype=np.int8)
    bits = ((byte_values[:, None] >> np.arange(7, -1, -1)) & 1).astype(np.int8)
    return bits.reshape(-1)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    bits = np.asarray(bits).reshape(-1)
    if bits.size == 0:
        return b""
    packed = np.packbits((bits > 0).astype(np.uint8), bitorder="big")
    return packed.tobytes()


def int_to_bit_list(x: int, bits: int = 16) -> np.ndarray:
    shifts = np.arange(bits - 1, -1, -1, dtype=np.int64)
    return ((x >> shifts) & 1).astype(np.int8)


def bits_to_int(bits: np.ndarray) -> int:
    bits = np.asarray(bits, dtype=np.int8).reshape(-1)
    if bits.size == 0:
        return 0
    bit_values = (bits > 0).astype(np.uint64)
    powers = np.arange(bits.size - 1, -1, -1, dtype=np.uint64)
    return int(np.sum(bit_values << powers))
