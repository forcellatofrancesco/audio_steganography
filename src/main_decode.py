from psk.psk_decoder import decode_from_audio
import tomllib

from psk.utils import load_waveform_from_file


def main():
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
        encoding_decoding_algorithm = config["algorithm"]["encoding_decoding"]

        waveform, sample_rate = load_waveform_from_file(config["input"]["waveform"])
        recovered_data = decode_from_audio(
            waveform,
            config["sync"]["preamble"],
            sample_rate=config["audio"]["sample_rate"],
            frequency=config["audio"]["frequency"],
            cycles_per_symbol=config["audio"]["cycles_per_symbol"],
            algorithm=encoding_decoding_algorithm,
        ).decode("utf-8", errors="replace")

        print(f"Sample rate: {sample_rate}")
        print(f"Recovered message:\n'{recovered_data}'")
        if config["debug"]["out_txt"]:
            with open(config["output"]["message"], "w") as out:
                print(recovered_data, file=out)


if __name__ == "__main__":
    main()
