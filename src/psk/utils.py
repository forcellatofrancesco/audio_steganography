import numpy as np
import os
import subprocess
import tempfile
import wave


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

    output_ext = os.path.splitext(filename)[1].lower()

    # Normalize to 16-bit range
    normalization = int(2**16 / 2 - 1)
    pcm_waveform = np.int16(np.clip(scaled, -1.0, 1.0) * normalization)

    def _write_wav_file(wav_path: str):
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16 bits
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_waveform.tobytes())

    if output_ext in {".wav", ".wave"}:
        _write_wav_file(filename)
        return

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        temp_wav_path = tmp_wav.name

    try:
        _write_wav_file(temp_wav_path)
        _convert_audio_via_ffmpeg(temp_wav_path, filename)
    finally:
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)


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
    source_path = filename
    temp_wav_path = None

    file_ext = os.path.splitext(filename)[1].lower()
    if file_ext not in {".wav", ".wave"}:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            temp_wav_path = tmp_wav.name
        _convert_audio_via_ffmpeg(filename, temp_wav_path)
        source_path = temp_wav_path

    try:
        with wave.open(source_path, "rb") as wf:
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
    finally:
        if temp_wav_path and os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)


def _convert_audio_via_ffmpeg(input_path: str, output_path: str):
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-ac",
        "1",
        output_path,
    ]

    try:
        subprocess.run(
            ffmpeg_cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg is required to convert non-WAV audio files. Please install ffmpeg."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffmpeg failed converting '{input_path}' to '{output_path}': {exc.stderr.strip()}"
        ) from exc
