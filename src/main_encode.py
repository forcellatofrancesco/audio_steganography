from psk.psk_encoder import (
    encode_to_audio,
)
import tomllib

from psk.utils import save_waveform_to_file
from voice_manager import get_voice_manager


def main():
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
        encoding_decoding_algorithm = config["algorithm"]["encoding_decoding"]
        with open(config["input"]["message"], "r") as input:
            message = "".join(input.readlines())
            preamble: list[int] = config["sync"]["preamble"]
            byte_data = message.encode("utf-8")
            waveform = encode_to_audio(
                byte_data,
                preamble,
                sample_rate=config["audio"]["sample_rate"],
                frequency=config["audio"]["frequency"],
                cycles_per_symbol=config["audio"]["cycles_per_symbol"],
                algorithm=encoding_decoding_algorithm,
            )
            save_waveform_to_file(
                waveform,
                config["output"]["waveform"],
                sample_rate=config["audio"]["sample_rate"],
                volume=config["audio"]["volume"],
            )
            print(f"Audio file created: {config['output']['waveform']}")
            manager = get_voice_manager(config["dataset"]["directory"])
            audios = manager.get_audios_by_total_duration(188.39)
            sum_durations = lambda ls: sum(map(lambda x: x[1], ls))
            print(sum_durations(audios))


if __name__ == "__main__":
    main()
