from psk.psk_encoder import (
    encode_to_audio,
)
import tomllib

from psk.utils import save_waveform_to_file


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


if __name__ == "__main__":
    main()
