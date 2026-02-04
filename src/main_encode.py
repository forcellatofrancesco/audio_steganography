from psk.psk_encoder import phase_shift_keying, save_waveform_to_file
import tomllib


def main():
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
        with open(config["input"]["message"], "r") as input:
            message = "\n".join(input.readlines())

            byte_data = message.encode("utf-8")
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
