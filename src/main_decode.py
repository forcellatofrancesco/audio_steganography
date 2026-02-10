from psk.psk_decoder import (
    align_to_start_sequence,
    decode_phase_shift_keying,
    load_waveform_from_file,
)
import tomllib


def main():
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
        with open(
            config["sync"]["start"],
            "r",
        ) as start_sequence_file, open(
            config["sync"]["end"],
            "r",
        ) as end_sequence_file:
            start_sequence = start_sequence_file.readline().rstrip("\n")
            end_sequence = end_sequence_file.readline().rstrip("\n")
            waveform, sample_rate = load_waveform_from_file(
                config["output"]["waveform"]
            )
            aligned_waveform = align_to_start_sequence(
                waveform,
                start_sequence,
                sample_rate=config["audio"]["sample_rate"],
                frequency=config["audio"]["frequency"],
                cycles_per_symbol=config["audio"]["cycles_per_symbol"],
            )
            recovered_data = decode_phase_shift_keying(
                aligned_waveform,
                sample_rate=config["audio"]["sample_rate"],
                frequency=config["audio"]["frequency"],
                cycles_per_symbol=config["audio"]["cycles_per_symbol"],
            ).decode("utf-8", errors="replace")
            start_index = recovered_data.find(start_sequence)
            if start_index != -1:
                start_index += len(start_sequence)
            else:
                start_index = 0
            end_index = recovered_data.find(end_sequence, start_index)
            if end_index == -1:
                end_index = len(recovered_data)
            message = recovered_data[start_index:end_index]
            print(f"Sample rate: {sample_rate}")
            print(f"Recovered message: '{message}'")
            with open(config["output"]["message"], "w") as out:
                print(message, file=out)


if __name__ == "__main__":
    main()
