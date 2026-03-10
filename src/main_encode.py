from audio.audiowaves import (
    apply_low_pass_filter,
    change_waveform_volume,
    generate_white_noise,
    load_and_concatenate_waveforms,
    save_waveform_to_file,
    sum_waveforms,
)
from audio.voice_manager import get_voice_manager
from psk.psk_encoder import (
    encode_to_audio,
)
import tomllib


def main():
    config = None
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
    encoding_decoding_algorithm = config["algorithm"]["encoding_decoding"]
    message = None
    with open(config["input"]["message"], "r") as input:
        message = "".join(input.readlines())
    preamble: list[int] = config["sync"]["preamble"]
    byte_data = message.encode("utf-8")
    data_wave, data_duration = encode_to_audio(
        byte_data,
        preamble,
        sample_rate=config["audio"]["sample_rate"],
        frequency=config["audio"]["frequency"],
        cycles_per_symbol=config["audio"]["cycles_per_symbol"],
        algorithm=encoding_decoding_algorithm,
    )
    # Remove harmonics in high frequencies
    data_wave = apply_low_pass_filter(
        data_wave,
        config["audio"]["high_pass_filter"],
        config["audio"]["sample_rate"],
    )
    data_wave = change_waveform_volume(data_wave, config["audio"]["volume_gain_data"])

    manager = get_voice_manager(config["dataset"]["directory"])
    audios = manager.get_audios_by_total_duration(data_duration)
    audios_waveform = load_and_concatenate_waveforms(
        list(map(lambda x: x[0], audios)),  # Remove durations
        config["audio"]["sample_rate"],
    )
    audios_waveform = change_waveform_volume(
        audios_waveform, config["audio"]["volume_gain_voice"]
    )

    white_noise = generate_white_noise(
        len(audios_waveform),
        stddev=config["audio"]["volume_noise"],
    )

    overlapped_waveforms = sum_waveforms(
        data_wave,
        audios_waveform,
        white_noise,
        random_offset=True,
    )
    save_waveform_to_file(
        overlapped_waveforms,
        config["output"]["waveform"],
        sample_rate=config["audio"]["sample_rate"],
    )
    print(f"Audio file created: {config['output']['waveform']}")


if __name__ == "__main__":
    main()
