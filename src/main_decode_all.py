import tomllib

import csv
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from collections import Counter
from audio.audiowaves import load_waveform_from_file
from psk.psk_decoder import decode_from_audio


def decode_row(row, preamble, cycles_per_symbol, encoding_decoding_algorithm):
    """Decode a single row's audio data."""
    waveform, sample_rate = load_waveform_from_file(row.get("download_path", ""))
    message = row.get("message", "")
    recovered_data, decode_status = decode_from_audio(
        waveform,
        preamble,
        sample_rate=sample_rate,
        frequency=int(float(row.get("frequency", -1.0))),
        cycles_per_symbol=cycles_per_symbol,
        algorithm=encoding_decoding_algorithm,
    )
    recovered_data = recovered_data.decode("utf-8", errors="replace")
    trimmed = recovered_data[: len(message)]
    difference = sum(1 for a, b in zip(trimmed, message) if a != b)
    error_rate = difference / len(message)
    return {
        **row,
        "error_rate": error_rate,
        "decoded_data": trimmed,
        "decode_status": decode_status,
    }


def main():
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
    encoding_decoding_algorithm = config["algorithm"]["encoding_decoding"]
    preamble = config["sync"]["preamble"]
    cycles_per_symbol = config["audio"]["cycles_per_symbol"]
    with open(
        config["output"]["automation_runs_csv"],
        "r",
        newline="",
        encoding="utf-8",
    ) as csv_input, open(
        config["output"]["decoded_csv"],
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_out:
        keys = [
            "frequency",
            "volume_gain_data",
            "message",
            "volume_noise",
            "download_path",
            "status",
            "error_rate",
            "decoded_data",
            "decode_status",
        ]
        reader = csv.DictReader(csv_input)
        writer = csv.DictWriter(csv_out, fieldnames=keys)
        writer.writeheader()

        rows = list(reader)
        path_counts = Counter(row.get("download_path", "") for row in rows)
        duplicated_paths = {
            path: count for path, count in path_counts.items() if path and count > 1
        }
        if duplicated_paths:
            print(
                "Warning: duplicate download_path values found. "
                "Some rows likely reference overwritten files. "
                f"Duplicated paths: {len(duplicated_paths)}"
            )

        total_rows = len(rows)
        last_printed_percentage = -1

        with ThreadPoolExecutor(max_workers=24) as executor:
            # map() automatically maintains order
            for idx, result in enumerate(
                executor.map(
                    decode_row,
                    rows,
                    repeat(preamble),
                    repeat(cycles_per_symbol),
                    repeat(encoding_decoding_algorithm),
                ),
                1,
            ):
                writer.writerow(result)

                # Print progress only when percentage point changes
                current_percentage = int(100 * idx / total_rows)
                if current_percentage > last_printed_percentage:
                    print(f"Progress: {current_percentage}% ({idx}/{total_rows})")
                    last_printed_percentage = current_percentage

            csv_out.flush()


if __name__ == "__main__":
    main()
