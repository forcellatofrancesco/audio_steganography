from audio.audiowaves import (
    change_waveform_volume,
    load_and_concatenate_waveforms,
    resample_waveform,
    save_waveform_to_file,
    sum_waveforms_with_overlap,
)
from audio.voice_manager import get_voice_manager
from psk.psk_encoder import (
    encode_to_audio,
)
import tomllib


def main():
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
        encoding_decoding_algorithm = config["algorithm"]["encoding_decoding"]
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
            data_wave = change_waveform_volume(
                data_wave, config["audio"]["volume_gain_data"]
            )
            manager = get_voice_manager(config["dataset"]["directory"])
            audios = manager.get_audios_by_total_duration(data_duration)
            audios_waveform, audios_sample_rate = load_and_concatenate_waveforms(
                list(map(lambda x: x[0], audios))
            )
            if audios_sample_rate is None:
                raise ValueError(
                    "Something went wrong with the concatenation of audio files."
                )
            audios_waveform = change_waveform_volume(
                audios_waveform, config["audio"]["volume_gain_voice"]
            )
            if audios_sample_rate != config["audio"]["sample_rate"]:
                audios_waveform = resample_waveform(
                    audios_waveform,
                    original_sample_rate=audios_sample_rate,
                    target_sample_rate=config["audio"]["sample_rate"],
                )
            overlapped_waveforms = sum_waveforms_with_overlap(
                data_wave,
                audios_waveform,
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
