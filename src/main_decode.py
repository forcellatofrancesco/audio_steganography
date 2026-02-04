from psk.psk_decoder import decode_phase_shift_keying, load_waveform_from_file
import tomllib


def main():
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
        waveform, sample_rate = load_waveform_from_file(config["output"]["waveform"])
        recovered_data = decode_phase_shift_keying(
            waveform,
            sample_rate=config["audio"]["sample_rate"],
            frequency=config["audio"]["frequency"],
            n=config["audio"]["n"],
        ).decode("utf-8")
        print(f"Sample rate: {sample_rate}")
        print(f"Recovered message: '{recovered_data}'")
        with open(config["output"]["message"], "w") as out:
            print(recovered_data, file=out)


if __name__ == "__main__":
    main()
