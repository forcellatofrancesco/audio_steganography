from psk.psk_decoder import decode_phase_shift_keying
import tomllib

from psk.utils import load_waveform_from_file


def main():
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
        waveform, sample_rate = load_waveform_from_file(config["input"]["waveform"])
        recovered_data = decode_phase_shift_keying(
            waveform,
            config["sync"]["preamble"],
            sample_rate=config["audio"]["sample_rate"],
            frequency=config["audio"]["frequency"],
            cycles_per_symbol=config["audio"]["cycles_per_symbol"],
        ).decode("utf-8", errors="replace")

        print(f"Sample rate: {sample_rate}")
        print(f"Recovered message: '{recovered_data}'")
        if config["debug"]["out_txt"]:
            with open(config["output"]["message"], "w") as out:
                print(recovered_data, file=out)


if __name__ == "__main__":
    main()
