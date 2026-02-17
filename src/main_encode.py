from psk.psk_encoder import (
    differential_binary_phase_shift_keying,
)
import tomllib

from psk.utils import save_waveform_to_file


def main():
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
        with open(config["input"]["message"], "r") as input:
            message = "".join(input.readlines())
            preamble: list[int] = config["sync"]["preamble"]
            byte_data = message.encode("utf-8")
            waveform = differential_binary_phase_shift_keying(
                byte_data,
                preamble,
                sample_rate=config["audio"]["sample_rate"],
                frequency=config["audio"]["frequency"],
                cycles_per_symbol=config["audio"]["cycles_per_symbol"],
            )
            save_waveform_to_file(
                waveform,
                config["output"]["waveform"],
                sample_rate=config["audio"]["sample_rate"],
                volume=config["audio"]["volume"],
            )
            print("Audio file created: output.wav")


if __name__ == "__main__":
    main()
