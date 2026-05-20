import tomllib

import csv
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from collections import Counter
from audio.audiowaves import load_waveform_from_file
from util.decoding import decode_and_score_message


def decode_row(row, preamble, cycles_per_symbol, encoding_decoding_algorithm):
    """Decode a single row's audio data."""
    waveform, sample_rate = load_waveform_from_file(row.get("download_path", ""))
    return {
        **row,
        **decode_and_score_message(
            waveform,
            sample_rate,
            row.get("message", ""),
            preamble,
            int(float(row.get("frequency", -1.0))),
            cycles_per_symbol,
            encoding_decoding_algorithm,
        ),
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
