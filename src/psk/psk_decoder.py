import numpy as np
import wave

from psk.psk_encoder import differential_binary_phase_shift_keying


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
    cycles_per_symbol=1.0,
    use_window=True,
    phase_offset=0.0,
):
    bits, _ = _extract_dbpsk_bits(
        waveform,
        sample_rate=sample_rate,
        frequency=frequency,
        cycles_per_symbol=cycles_per_symbol,
        use_window=use_window,
        phase_offset=phase_offset,
    )
    if not bits:
        return b""

    bytes_out: list[int] = []
    for i in range(0, len(bits) - (len(bits) % 8), 8):
        value = 0
        for bit in bits[i : i + 8]:
            value = (value << 1) | bit
        bytes_out.append(value)

    return bytes(bytes_out)


def _extract_dbpsk_bits(
    waveform,
    sample_rate,
    frequency,
    cycles_per_symbol,
    use_window=True,
    phase_offset=0.0,
    sample_start=0,
):
    if waveform.size == 0:
        return [], 0

    if frequency <= 0:
        return [], 0

    if cycles_per_symbol <= 0:
        return [], 0

    samples_per_symbol = max(1, int(round(sample_rate * cycles_per_symbol / frequency)))
    if sample_start < 0 or sample_start >= waveform.size:
        return [], samples_per_symbol

    total_symbols = (waveform.size - sample_start) // samples_per_symbol
    if total_symbols <= 0:
        return [], samples_per_symbol

    phases = []
    sample_offset = 0
    window = np.hanning(samples_per_symbol) if use_window else None

    for symbol_index in range(total_symbols):
        start = sample_start + symbol_index * samples_per_symbol
        end = start + samples_per_symbol
        segment = waveform[start:end]

        if segment.size == 0:
            break

        t_symbol = (np.arange(samples_per_symbol) + sample_offset) / sample_rate
        reference = np.exp(-1j * 2 * np.pi * frequency * t_symbol)
        if use_window:
            segment = segment * window
            reference = reference * window

        correlation = np.sum(segment * reference)
        phases.append(np.angle(correlation) + phase_offset)
        sample_offset += samples_per_symbol

    if not phases:
        return [], samples_per_symbol

    bits = []
    first_phase = phases[0]
    first_bit = 0 if np.cos(first_phase) >= 0 else 1
    bits.append(first_bit)
    previous_bit = first_bit

    for index in range(1, len(phases)):
        phase_diff = np.angle(np.exp(1j * (phases[index] - phases[index - 1])))
        phase_toggled = np.cos(phase_diff) < 0
        if phase_toggled:
            previous_bit ^= 1
        bits.append(previous_bit)

    return bits, samples_per_symbol


def align_to_start_sequence(
    waveform,
    start_sequence,
    sample_rate,
    frequency,
    cycles_per_symbol,
):
    start_bytes = start_sequence.encode("utf-8")
    start_bits = []
    for byte in start_bytes:
        for bit_index in range(7, -1, -1):
            start_bits.append((byte >> bit_index) & 1)

    if not start_bits:
        return waveform

    samples_per_symbol = max(1, int(round(sample_rate * cycles_per_symbol / frequency)))
    inverted_start_bits = [1 - bit for bit in start_bits]

    best_offset = None
    best_index = None
    best_score = -1
    for sample_start in range(samples_per_symbol):
        bits, _ = _extract_dbpsk_bits(
            waveform,
            sample_rate=sample_rate,
            frequency=frequency,
            cycles_per_symbol=cycles_per_symbol,
            sample_start=sample_start,
        )
        if not bits or len(bits) < len(start_bits):
            continue

        max_start = len(bits) - len(start_bits)
        for i in range(max_start + 1):
            window = bits[i : i + len(start_bits)]
            if window == start_bits or window == inverted_start_bits:
                best_offset = sample_start
                best_index = i
                best_score = len(start_bits)
                break
        if best_score == len(start_bits):
            break

    if best_offset is None or best_index is None:
        return waveform

    offset = best_offset + best_index * samples_per_symbol
    return waveform[offset:]


# def find_sync_offset(waveform, sync_waveform):
#     if waveform.size == 0 or sync_waveform.size == 0:
#         return None

#     if waveform.size < sync_waveform.size:
#         return None

#     sync_norm = np.linalg.norm(sync_waveform)
#     if sync_norm == 0:
#         return None

#     # Normalize sync waveform to avoid amplitude bias during correlation.
#     sync_unit = sync_waveform / sync_norm
#     correlations = np.correlate(waveform, sync_unit, mode="valid")
#     if correlations.size == 0:
#         return None

#     return int(np.argmax(correlations))


# def align_to_start_sequence(
#     waveform,
#     start_sequence,
#     sample_rate,
#     frequency,
#     cycles_per_symbol,
# ):
#     start_bytes = start_sequence.encode("utf-8")
#     start_bits = []
#     for byte in start_bytes:
#         for bit_index in range(7, -1, -1):
#             start_bits.append((byte >> bit_index) & 1)

#     bits, samples_per_symbol = _extract_dbpsk_bits(
#         waveform,
#         sample_rate=sample_rate,
#         frequency=frequency,
#         cycles_per_symbol=cycles_per_symbol,
#     )
#     if not bits or not start_bits:
#         return waveform

#     inverted_start_bits = [1 - bit for bit in start_bits]
#     max_start = len(bits) - len(start_bits)
#     if max_start < 0:
#         return waveform

#     match_index = None
#     for i in range(max_start + 1):
#         window = bits[i : i + len(start_bits)]
#         if window == start_bits or window == inverted_start_bits:
#             match_index = i
#             break

#     if match_index is None:
#         return waveform

#     offset = match_index * samples_per_symbol
#     return waveform[offset:]
