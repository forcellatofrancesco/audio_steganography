from psk.psk_encoder import phase_shift_keying, save_waveform_to_file
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
            waveform = phase_shift_keying(
                byte_data,
                sample_rate=config["audio"]["sample_rate"],
                frequency=config["audio"]["frequency"],
                n=config["audio"]["n"],
            )
            save_waveform_to_file(waveform, config["output"]["waveform"])
            print("Audio file created: output.wav")


if __name__ == "__main__":
    main()
