import numpy as np


def bits_to_phase_wave(
    bits: np.ndarray,
    frequency: int,
    cycles_per_symbol: float,
    sample_rate: int,
) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.int8).reshape(-1)
    samples_per_symbol = max(1, int(round(sample_rate * cycles_per_symbol / frequency)))
    total_samples = bits.size * samples_per_symbol
    if total_samples == 0:
        return np.empty(0, dtype=np.float64)

    time_vector = np.arange(total_samples, dtype=np.float64) / sample_rate
    phase_vector = np.repeat(np.pi * bits.astype(np.float64), samples_per_symbol)
    return np.sin(2 * np.pi * frequency * time_vector + phase_vector)


def bytes_to_bits(data: bytes) -> np.ndarray:
    byte_values = np.frombuffer(data, dtype=np.uint8)
    if byte_values.size == 0:
        return np.array([], dtype=np.int8)
    bits = ((byte_values[:, None] >> np.arange(7, -1, -1)) & 1).astype(np.int8)
    return bits.reshape(-1)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    if bits.size == 0:
        return b""
    packed = np.packbits((bits > 0).astype(np.uint8), bitorder="big")
    usable_len = bits.size // 8
    if usable_len == 0:
        return b""
    return packed[:usable_len].tobytes()


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
