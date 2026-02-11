import numpy as np
import wave


def bit_to_phase_wave(
    bit: int, frequency, samples_per_symbol, sample_offset, sample_rate
):
    phase = np.pi * bit  # phase = 0 when xored = 0, phase = pi when xored = 1
    t_symbol = (np.arange(samples_per_symbol) + sample_offset) / sample_rate
    symbol = np.sin(2 * np.pi * frequency * t_symbol + phase)
    return symbol


def differential_binary_phase_shift_keying(
    data: bytes, sample_rate=44100, frequency=440, cycles_per_symbol=1.0
):
    if frequency <= 0:
        raise ValueError("frequency must be positive.")
    if cycles_per_symbol <= 0:
        raise ValueError("cycles_per_symbol must be positive.")
    samples_per_symbol = max(1, int(round(sample_rate * cycles_per_symbol / frequency)))
    waveform_chunks = []
    previous_bit = 1
    waveform_chunks.append(
        bit_to_phase_wave(
            previous_bit,
            frequency,
            samples_per_symbol,
            0,
            sample_rate,
        )
    )
    sample_offset = samples_per_symbol
    original_str = ""
    xored_str = "1"
    for byte in data:
        for bit_index in range(7, -1, -1):
            bit = (byte >> bit_index) & 1
            xored = previous_bit ^ bit
            original_str += str(bit)
            xored_str += str(xored)
            previous_bit = xored
            symbol = bit_to_phase_wave(
                xored,
                frequency,
                samples_per_symbol,
                sample_offset,
                sample_rate,
            )
            waveform_chunks.append(symbol)
            sample_offset += samples_per_symbol
    return np.concatenate(waveform_chunks) if waveform_chunks else np.array([])


def differential_binary_phase_shift_keying_deprecated(
    data: bytes, sample_rate=44100, frequency=440, cycles_per_symbol=1.0
):
    # DBPSK: toggle phase by pi when the bit changes
    if frequency <= 0:
        raise ValueError("frequency must be positive.")
    if cycles_per_symbol <= 0:
        raise ValueError("cycles_per_symbol must be positive.")

    samples_per_symbol = max(1, int(round(sample_rate * cycles_per_symbol / frequency)))
    waveform_chunks = []
    previous_bit = None
    phase = 0.0
    sample_offset = 0
    for byte in data:
        for bit_index in range(7, -1, -1):
            bit = (byte >> bit_index) & 1
            if previous_bit is None:
                previous_bit = bit
                phase = 0.0 if bit == 0 else np.pi
            elif previous_bit != bit:
                previous_bit = bit
                phase += np.pi
            t_symbol = (np.arange(samples_per_symbol) + sample_offset) / sample_rate
            symbol = np.sin(2 * np.pi * frequency * t_symbol + phase)
            waveform_chunks.append(symbol)
            sample_offset += samples_per_symbol

    return np.concatenate(waveform_chunks) if waveform_chunks else np.array([])


def save_waveform_to_file(waveform, filename, sample_rate=44100, volume=1.0):
    """
    Save a waveform array to a WAV file with 16-bit mono audio encoding.

    Args:
        waveform (np.ndarray): A numpy array containing audio samples.
            Values should be in the range [-1, 1] for optimal normalization.
        filename (str): The output file path where the WAV file will be saved.
        sample_rate (int, optional): The sample rate in Hz. Defaults to 44100 Hz (CD quality).
        volume (float, optional): Gain multiplier in [0.0, 1.0]. Defaults to 1.0.

    Returns:
        None

    Raises:
        ValueError: If the waveform array is empty.
        IOError: If the file cannot be written to the specified path.

    Notes:
        - The waveform is only normalized if it exceeds the [-1, 1] range.
        - The volume is applied after any normalization.
        - The output is mono (single channel).
        - The sample width is fixed at 2 bytes (16 bits).

    Example:
        >>> waveform = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        >>> save_waveform_to_file(waveform, "tone.wav", sample_rate=44100)
    """
    if waveform.size == 0:
        raise ValueError("Waveform array is empty.")

    volume = float(volume)
    if volume < 0.0 or volume > 1.0:
        raise ValueError("volume must be in the range [0.0, 1.0].")

    peak = np.max(np.abs(waveform))
    if peak == 0:
        scaled = waveform
    elif peak > 1.0:
        scaled = (waveform / peak) * volume
    else:
        scaled = waveform * volume

    # Normalize to 16-bit range
    normalization = int(2**16 / 2 - 1)
    waveform = np.int16(np.clip(scaled, -1.0, 1.0) * normalization)
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16 bits
        wf.setframerate(sample_rate)
        wf.writeframes(waveform.tobytes())
