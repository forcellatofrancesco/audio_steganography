import numpy as np
import wave

from .psk_encoder import (
    bit_to_phase_wave,
    encode_dbpsk_list,
)


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


def decode(bits: list[int]) -> list[int]:
    previous_bit = bits[0]
    xored_bits = []
    for bit in bits[1:]:
        xored = bit ^ previous_bit
        previous_bit = bit
        xored_bits.append(xored)
    return xored_bits


def bits_to_bytes(bits: list[int]) -> bytes:
    byte_values = []
    current = 0
    for idx, bit in enumerate(bits):
        current = (current << 1) | bit
        if (idx + 1) % 8 == 0:
            byte_values.append(current)
            current = 0
    return bytes(byte_values)


def detect_preamble(
    waveform,
    preamble,
    sample_rate=44100,
    frequency=440,
    cycles_per_symbol=1.0,
):
    """
    Find the sample offset of a DBPSK preamble using matched filtering.

    Args:
            waveform (np.ndarray): Input audio samples.
            preamble (list[int]): Preamble bits before differential encoding.
            sample_rate (int): Sample rate in Hz.
            frequency (float): Carrier frequency in Hz.
            cycles_per_symbol (float): Carrier cycles per symbol.

    Returns:
            int: Sample index of the best preamble match.
    """
    if waveform.size == 0:
        return 0

    samples_per_symbol = max(1, int(round(sample_rate * cycles_per_symbol / frequency)))
    encoded_preamble = encode_dbpsk_list(preamble)

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
        return 0

    expected = expected - np.mean(expected)
    candidate = waveform - np.mean(waveform)
    correlation = np.correlate(candidate, expected, mode="valid")
    if correlation.size == 0:
        return 0

    return int(np.argmax(np.abs(correlation)))


def decode_phase_shift_keying(
    waveform,
    preamble: list[int],
    sample_rate=44100,
    frequency=440,
    cycles_per_symbol=1.0,
):
    if frequency <= 0:
        raise ValueError("frequency must be positive.")
    if cycles_per_symbol <= 0:
        raise ValueError("cycles_per_symbol must be positive.")
    start_index = detect_preamble(
        waveform,
        preamble,
        sample_rate=sample_rate,
        frequency=frequency,
        cycles_per_symbol=cycles_per_symbol,
    )
    trimmed_waveform = waveform[start_index:]

    samples_per_symbol = max(1, int(round(sample_rate * cycles_per_symbol / frequency)))
    if trimmed_waveform.size < samples_per_symbol:
        return b""

    bits = []
    for start in range(
        0, trimmed_waveform.size - samples_per_symbol + 1, samples_per_symbol
    ):
        phase = _symbol_phase(
            trimmed_waveform, start, samples_per_symbol, sample_rate, frequency
        )
        if phase is None:
            break
        bit = 0 if np.cos(phase) >= 0 else 1
        bits.append(bit)
    decoded_bits = decode(bits)
    # Strip preamble
    return bits_to_bytes(decoded_bits[len(preamble) :])
