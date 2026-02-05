import numpy as np
import wave

from psk.psk_encoder import phase_shift_keying


def load_waveform_from_file(filename):
    """
    Load a WAV file and return a mono waveform normalized to [-1, 1].

    Args:
            filename (str): Path to the WAV file.

    Returns:
            tuple[np.ndarray, int]: (waveform, sample_rate)

    Raises:
            ValueError: If the WAV file has an unsupported sample width.
            IOError: If the file cannot be read.
    """
    with wave.open(filename, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        data = data / 32768.0
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
        data = data / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth} bytes")

    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)

    return data, sample_rate


def decode_phase_shift_keying(
    waveform,
    sample_rate=44100,
    frequency=440.0,
    n=1,
    use_window=True,
    phase_offset=np.pi / 2,
):
    if waveform.size == 0:
        return b""

    samples_per_symbol = sample_rate // 8
    if samples_per_symbol <= 0:
        return b""

    symbol_len = samples_per_symbol * max(1, int(n))
    total_symbols = waveform.size // symbol_len
    if total_symbols <= 0:
        return b""
    byte_values = []
    t_symbol = np.arange(samples_per_symbol) / sample_rate
    reference = np.exp(-1j * 2 * np.pi * frequency * t_symbol)
    if use_window:
        window = np.hanning(samples_per_symbol)
        reference = reference * window

    for symbol_index in range(total_symbols):
        start = symbol_index * symbol_len
        end = start + symbol_len
        symbol_chunk = waveform[start:end]

        correlations = []
        for repeat in range(max(1, int(n))):
            seg_start = repeat * samples_per_symbol
            seg_end = seg_start + samples_per_symbol
            segment = symbol_chunk[seg_start:seg_end]

            if segment.size == 0:
                continue

            if use_window:
                segment = segment * np.hanning(segment.size)

            correlation = np.sum(segment * reference)
            correlations.append(correlation)

        if not correlations:
            byte_values.append(0)
            continue

        avg_correlation = np.mean(correlations)
        avg_phase = np.angle(avg_correlation) + phase_offset
        phase_norm = avg_phase % (2 * np.pi)
        byte_value = int(np.round(phase_norm / (2 * np.pi) * 255)) % 256
        byte_values.append(byte_value)

    return bytes(byte_values)


def find_sync_offset(waveform, sync_waveform):
    if waveform.size == 0 or sync_waveform.size == 0:
        return None

    if waveform.size < sync_waveform.size:
        return None

    sync_norm = np.linalg.norm(sync_waveform)
    if sync_norm == 0:
        return None

    # Normalize sync waveform to avoid amplitude bias during correlation.
    sync_unit = sync_waveform / sync_norm
    correlations = np.correlate(waveform, sync_unit, mode="valid")
    if correlations.size == 0:
        return None

    return int(np.argmax(correlations))


def align_to_start_sequence(waveform, start_sequence, sample_rate, frequency, n):
    sync_waveform = phase_shift_keying(
        start_sequence.encode("utf-8"),
        sample_rate=sample_rate,
        frequency=frequency,
        n=n,
    )
    offset = find_sync_offset(waveform, sync_waveform)
    if offset is None:
        return waveform

    return waveform[offset:]
