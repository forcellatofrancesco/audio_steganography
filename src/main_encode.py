from psk.psk_encoder import (
    differential_binary_phase_shift_keying,
    differential_binary_phase_shift_keying_deprecated,
    save_waveform_to_file,
)
import tomllib


def main():
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)

        with open(
            config["input"]["message"],
            "r",
        ) as input, open(
            config["sync"]["start"],
            "r",
        ) as start_sequence_file, open(
            config["sync"]["end"],
            "r",
        ) as end_sequence_file:
            message = "".join(input.readlines())
            start_sequence = start_sequence_file.readline().rstrip("\n")
            end_sequence = end_sequence_file.readline().rstrip("\n")

            byte_data = (start_sequence + message + end_sequence).encode("utf-8")
            waveform = differential_binary_phase_shift_keying(
                byte_data,
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
