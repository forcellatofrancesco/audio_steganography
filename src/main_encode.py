from audio.audiowaves import (
    apply_low_pass_filter,
    change_waveform_volume,
    generate_white_noise,
    load_and_concatenate_waveforms,
    save_waveform_to_file,
    sum_waveforms,
)
from audio.voice_manager import get_voice_manager
from audio.audio_config import AudioConfigFactory, AudioConfig
import numpy as np
from psk.psk_encoder import (
    encode_to_audio,
)
import tomllib


def encode_from_config(
    byte_data: bytes,
    preamble: np.ndarray,
    algorithm: str,
    dataset_directory: str,
    output_path: str,
    seed: int | None,
    audio_config: AudioConfig,
) -> None:
    data_wave, data_duration = encode_to_audio(
        byte_data,
        preamble,
        sample_rate=audio_config.sample_rate,
        frequency=audio_config.frequency,
        cycles_per_symbol=audio_config.cycles_per_symbol,
        algorithm=algorithm,
    )
    # Remove harmonics in high frequencies
    data_wave = apply_low_pass_filter(
        data_wave,
        audio_config.high_pass_filter,
        audio_config.sample_rate,
    )
    data_wave = change_waveform_volume(data_wave, audio_config.volume_gain_data)

    manager = get_voice_manager(dataset_directory)
    audios = manager.get_audios_by_total_duration(
        data_duration,
        shuffled=True,
        seed=seed,
    )
    audios_waveform = load_and_concatenate_waveforms(
        list(map(lambda x: x[0], audios)),  # Remove durations
        audio_config.sample_rate,
    )
    audios_waveform = change_waveform_volume(
        audios_waveform, audio_config.volume_gain_voice
    )

    white_noise = generate_white_noise(
        len(audios_waveform),
        stddev=audio_config.volume_noise,
        seed=seed,
    )

    overlapped_waveforms = sum_waveforms(
        data_wave,
        audios_waveform,
        white_noise,
        random_offset=True,
    )
    save_waveform_to_file(
        overlapped_waveforms,
        output_path,
        sample_rate=audio_config.sample_rate,
    )


def main():
    config = None
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
    audio_config = AudioConfigFactory(data=config).create()
    encoding_decoding_algorithm = config["algorithm"]["encoding_decoding"]
    message = None
    with open(config["input"]["message"], "r") as input:
        message = "".join(input.readlines())
    preamble = np.asarray(config["sync"]["preamble"], dtype=np.int8)
    byte_data = message.encode("utf-8")
    encode_from_config(
        byte_data,
        preamble,
        encoding_decoding_algorithm,
        config["dataset"]["directory"],
        config["output"]["waveform"],
        config["algorithm"].get("shuffle_seed"),
        audio_config,
    )
    print(f"Audio file created: {config['output']['waveform']}")


if __name__ == "__main__":
    main()
