import numpy as np
import wave

from .psk_encoder import differential_binary_phase_shift_keying, encode_dbpsk


def load_waveform_from_file(filename):
    """
    Load a WAV file and return normalized mono waveform and sample rate.

    Args:
            filename (str): Path to the WAV file.

    Returns:
            tuple[np.ndarray, int]: (waveform, sample_rate)

    Raises:
            ValueError: If the WAV format is unsupported.
            IOError: If the file cannot be read.
    """
    with wave.open(filename, "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frame_count = wf.getnframes()
        raw = wf.readframes(frame_count)

    if sample_width == 1:
        dtype = np.uint8
        data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sample_width == 2:
        dtype = np.int16
        data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        data = data / 32768.0
    elif sample_width == 4:
        dtype = np.int32
        data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        data = data / 2147483648.0
    else:
        raise ValueError("Unsupported sample width: %s" % sample_width)

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)

    return data, sample_rate


def _symbol_phase(waveform, start, samples_per_symbol, sample_rate, frequency):
    segment = waveform[start : start + samples_per_symbol]
    if segment.size < samples_per_symbol:
        return None
    t = (np.arange(samples_per_symbol) + start) / sample_rate
    sin_ref = np.sin(2 * np.pi * frequency * t)
    cos_ref = np.cos(2 * np.pi * frequency * t)
    i = np.dot(segment, sin_ref)
    q = np.dot(segment, cos_ref)
    return np.arctan2(q, i)


def decode(bits: list[int]) -> bytes:
    previous_bit = bits[0]
    xored_bits = []
    for bit in bits[1:]:
        xored = bit ^ previous_bit
        previous_bit = bit
        xored_bits.append(xored)
    byte_values = []
    current = 0
    for idx, bit in enumerate(xored_bits):
        current = (current << 1) | bit
        if (idx + 1) % 8 == 0:
            byte_values.append(current)
            current = 0

    return bytes(byte_values)


def search_start(bits: list[int], start_sequence: str):
    start: list[int] = encode_dbpsk(start_sequence.encode("utf-8"))
    # Create sliding windows of length len(b)
    windows = np.lib.stride_tricks.sliding_window_view(bits, len(start))

    # Compare each window with b
    matches = np.all(windows == start, axis=1)

    # Get first occurrence
    indices = np.where(matches)[0]
    if len(indices) == 0:
        return None
    return indices[0]


def decode_phase_shift_keying(
    waveform,
    start_sequence: str,
    sample_rate=44100,
    frequency=440,
    cycles_per_symbol=1.0,
):
    if frequency <= 0:
        raise ValueError("frequency must be positive.")
    if cycles_per_symbol <= 0:
        raise ValueError("cycles_per_symbol must be positive.")

    samples_per_symbol = max(1, int(round(sample_rate * cycles_per_symbol / frequency)))
    if waveform.size < samples_per_symbol:
        return b""

    bits = []
    for start in range(0, waveform.size - samples_per_symbol + 1, samples_per_symbol):
        phase = _symbol_phase(
            waveform, start, samples_per_symbol, sample_rate, frequency
        )
        if phase is None:
            break
        bit = 0 if np.cos(phase) >= 0 else 1
        bits.append(bit)
    start_index = search_start(bits, start_sequence)
    return decode(bits[start_index:])


def decode_phase_shift_keying_not_aligned(
    waveform, sample_rate=44100, frequency=440, cycles_per_symbol=1.0
):
    if frequency <= 0:
        raise ValueError("frequency must be positive.")
    if cycles_per_symbol <= 0:
        raise ValueError("cycles_per_symbol must be positive.")

    samples_per_symbol = max(1, int(round(sample_rate * cycles_per_symbol / frequency)))
    if waveform.size < samples_per_symbol:
        return b""

    bits = []
    for start in range(0, waveform.size - samples_per_symbol + 1, samples_per_symbol):
        phase = _symbol_phase(
            waveform, start, samples_per_symbol, sample_rate, frequency
        )
        if phase is None:
            break
        bit = 0 if np.cos(phase) >= 0 else 1
        bits.append(bit)
    previous_bit = bits[0]
    xored_bits = []
    for bit in bits[1:]:
        xored = bit ^ previous_bit
        previous_bit = bit
        xored_bits.append(xored)
    byte_values = []
    current = 0
    for idx, bit in enumerate(xored_bits):
        current = (current << 1) | bit
        if (idx + 1) % 8 == 0:
            byte_values.append(current)
            current = 0

    return bytes(byte_values)


def decode_phase_shift_keying_deprecated(
    waveform, sample_rate=44100, frequency=440, cycles_per_symbol=1.0
):
    """
    Decode a DBPSK waveform into bytes.

    Args:
            waveform (np.ndarray): Audio samples.
            sample_rate (int): Sample rate in Hz.
            frequency (float): Carrier frequency in Hz.
            cycles_per_symbol (float): Carrier cycles per symbol.

    Returns:
            bytes: Decoded byte stream.
    """
    if frequency <= 0:
        raise ValueError("frequency must be positive.")
    if cycles_per_symbol <= 0:
        raise ValueError("cycles_per_symbol must be positive.")

    samples_per_symbol = max(1, int(round(sample_rate * cycles_per_symbol / frequency)))
    if waveform.size < samples_per_symbol:
        return b""

    bits = []
    for start in range(0, waveform.size - samples_per_symbol + 1, samples_per_symbol):
        phase = _symbol_phase(
            waveform, start, samples_per_symbol, sample_rate, frequency
        )
        if phase is None:
            break
        bit = 0 if np.cos(phase) >= 0 else 1
        bits.append(bit)

    byte_values = []
    current = 0
    for idx, bit in enumerate(bits):
        current = (current << 1) | bit
        if (idx + 1) % 8 == 0:
            byte_values.append(current)
            current = 0

    return bytes(byte_values)


def align_to_start_sequence(
    waveform,
    start_sequence,
    sample_rate=44100,
    frequency=440,
    cycles_per_symbol=1.0,
):
    """
    Align waveform to the start sequence by matched filtering.

    Args:
            waveform (np.ndarray): Input audio samples.
            start_sequence (str): Text sequence used as preamble.
            sample_rate (int): Sample rate in Hz.
            frequency (float): Carrier frequency in Hz.
            cycles_per_symbol (float): Carrier cycles per symbol.

    Returns:
            np.ndarray: Waveform starting at the best match.
    """
    if waveform.size == 0:
        return waveform

    expected = differential_binary_phase_shift_keying(
        start_sequence.encode("utf-8"),
        sample_rate=sample_rate,
        frequency=frequency,
        cycles_per_symbol=cycles_per_symbol,
    )
    if expected.size == 0 or waveform.size < expected.size:
        return waveform

    expected = expected - np.mean(expected)
    candidate = waveform - np.mean(waveform)
    correlation = np.correlate(candidate, expected, mode="valid")
    if correlation.size == 0:
        return waveform

    best_offset = int(np.argmax(np.abs(correlation)))
    return waveform[best_offset:]
