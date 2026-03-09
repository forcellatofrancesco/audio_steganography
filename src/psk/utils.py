import numpy as np
from scipy.signal import resample_poly
import os
import subprocess
import tempfile
import wave
import math


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
        - The waveform is only normalized if it exceeds the [-1, 1] range.
        - The output is mono (single channel).
        - The sample width is fixed at 2 bytes (16 bits).

    Example:
        >>> waveform = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        >>> save_waveform_to_file(waveform, "tone.wav", sample_rate=44100)
    """
    if waveform.size == 0:
        raise ValueError("Waveform array is empty.")

    peak = np.max(np.abs(waveform))
    if peak == 0:
        scaled = waveform
    elif peak > 1.0:
        scaled = waveform / peak
    else:
        scaled = waveform

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


def change_waveform_volume(waveform: np.ndarray, gain: float) -> np.ndarray:
    """
    Scale a mono waveform by a gain factor.

    Args:
            waveform (np.ndarray): Input 1D waveform.
            gain (float): Non-negative gain multiplier.
                    ``1.0`` keeps the waveform unchanged, ``0.0`` silences it.

    Returns:
            np.ndarray: Gain-adjusted waveform (float32).

    Raises:
            ValueError: If waveform is not 1D, or gain is negative/non-finite.
    """
    waveform = np.asarray(waveform)
    if waveform.ndim != 1:
        raise ValueError("waveform must be a 1D array.")

    gain = float(gain)
    if not np.isfinite(gain) or gain < 0.0:
        raise ValueError("gain must be a finite, non-negative number.")

    if waveform.size == 0:
        return np.array([], dtype=np.float32)

    return (waveform.astype(np.float32) * gain).astype(np.float32, copy=False)


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


def load_and_concatenate_waveforms(file_paths: list[str]):
    """
    Load multiple FLAC files and concatenate them into one mono waveform.

    Args:
            file_paths (list[str]): Ordered list of ``.flac`` file paths.

    Returns:
            tuple[np.ndarray, int]: (concatenated_waveform, sample_rate)

    Raises:
            ValueError: If file list is empty, contains non-FLAC files, or sample rates differ.
    """
    if not file_paths:
        raise ValueError("file_paths must contain at least one FLAC file path.")

    waveform_chunks = []
    sample_rate = None

    for path in file_paths:
        file_ext = os.path.splitext(path)[1].lower()
        if file_ext != ".flac":
            raise ValueError(f"Expected a .flac file, got: {path}")

        waveform, current_sample_rate = load_waveform_from_file(path)
        if sample_rate is None:
            sample_rate = current_sample_rate
        elif current_sample_rate != sample_rate:
            raise ValueError(
                f"Sample rate mismatch for '{path}': expected {sample_rate}, got {current_sample_rate}."
            )

        waveform_chunks.append(waveform)

    concatenated = np.concatenate(waveform_chunks)
    return concatenated, sample_rate


def resample_waveform(
    waveform: np.ndarray,
    original_sample_rate: int,
    target_sample_rate: int,
) -> np.ndarray:
    """
    Resample a mono waveform to a new sample rate.

    Args:
            waveform (np.ndarray): Input 1D waveform.
            original_sample_rate (int): Source sample rate in Hz.
            target_sample_rate (int): Target sample rate in Hz.

    Returns:
            np.ndarray: Resampled waveform (float32).

    Raises:
            ValueError: If waveform shape is invalid or sample rates are not positive.
    """
    if original_sample_rate <= 0 or target_sample_rate <= 0:
        raise ValueError("Sample rates must be positive integers.")

    waveform = np.asarray(waveform)
    if waveform.ndim != 1:
        raise ValueError("waveform must be a 1D array.")
    if waveform.size == 0:
        return waveform.astype(np.float32)
    if original_sample_rate == target_sample_rate:
        return waveform.astype(np.float32, copy=True)

    common_divisor = math.gcd(original_sample_rate, target_sample_rate)
    up = target_sample_rate // common_divisor
    down = original_sample_rate // common_divisor

    resampled = resample_poly(waveform, up=up, down=down)
    return resampled.astype(np.float32)


def sum_waveforms_with_overlap(
    waveform_a: np.ndarray,
    waveform_b: np.ndarray,
    waveform_b_start: int = 0,
) -> np.ndarray:
    """
    Sum two mono waveforms so they overlap in time.

    Args:
            waveform_a (np.ndarray): First input waveform.
            waveform_b (np.ndarray): Second input waveform.
            waveform_b_start (int): Start index of ``waveform_b`` relative to
                    ``waveform_a`` in samples. ``0`` means both start together.
                    Positive values delay ``waveform_b``; negative values start it earlier.

    Returns:
            np.ndarray: The mixed waveform as float32.

    Raises:
            ValueError: If either waveform is not 1D.
            TypeError: If ``waveform_b_start`` is not an integer.
    """
    if not isinstance(waveform_b_start, (int, np.integer)):
        raise TypeError("waveform_b_start must be an integer sample index.")

    waveform_a = np.asarray(waveform_a, dtype=np.float32)
    waveform_b = np.asarray(waveform_b, dtype=np.float32)

    if waveform_a.ndim != 1 or waveform_b.ndim != 1:
        raise ValueError("Both waveforms must be 1D arrays.")

    if waveform_a.size == 0 and waveform_b.size == 0:
        return np.array([], dtype=np.float32)

    anchor = min(0, int(waveform_b_start))
    offset_a = -anchor
    offset_b = int(waveform_b_start) - anchor
    total_samples = max(offset_a + waveform_a.size, offset_b + waveform_b.size)

    mixed = np.zeros(total_samples, dtype=np.float32)
    mixed[offset_a : offset_a + waveform_a.size] += waveform_a
    mixed[offset_b : offset_b + waveform_b.size] += waveform_b
    return mixed


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
