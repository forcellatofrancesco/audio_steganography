import numpy as np
import wave


def phase_shift_keying(data, sample_rate=44100, frequency=440, n=1):
    # Create a waveform where each byte's symbol is repeated n times
    samples_per_symbol = sample_rate // 8
    t_symbol = np.arange(samples_per_symbol) / sample_rate
    waveform_chunks = []

    for byte in data:
        # Create a phase shift for each byte (symbol)
        phase = (byte / 255) * 2 * np.pi
        print(phase, end=";")
        symbol = np.sin(2 * np.pi * frequency * t_symbol + phase)
        waveform_chunks.append(np.tile(symbol, n))
    print("")

    return np.concatenate(waveform_chunks) if waveform_chunks else np.array([])


def save_waveform_to_file(waveform, filename, sample_rate=44100):
    """
    Save a waveform array to a WAV file with 16-bit mono audio encoding.

    Args:
        waveform (np.ndarray): A numpy array containing audio samples.
            Values should be in the range [-1, 1] for optimal normalization.
        filename (str): The output file path where the WAV file will be saved.
        sample_rate (int, optional): The sample rate in Hz. Defaults to 44100 Hz (CD quality).

    Returns:
        None

    Raises:
        ValueError: If the waveform array is empty.
        IOError: If the file cannot be written to the specified path.

    Notes:
        - The waveform is normalized to the 16-bit signed integer range [-32768, 32767].
        - The output is mono (single channel).
        - The sample width is fixed at 2 bytes (16 bits).

    Example:
        >>> waveform = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        >>> save_waveform_to_file(waveform, "tone.wav", sample_rate=44100)
    """
    # Normalize to 16-bit range
    normalization = int(2**16 / 2 - 1)
    waveform = np.int16((waveform / np.max(np.abs(waveform))) * normalization)
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16 bits
        wf.setframerate(sample_rate)
        wf.writeframes(waveform.tobytes())


def main():
    input_string = "hello wooooorld"
    byte_data = input_string.encode("utf-8")
    waveform = phase_shift_keying(byte_data)
    save_waveform_to_file(waveform, "output.wav")
    print("Audio file created: output.wav")


if __name__ == "__main__":
    main()
