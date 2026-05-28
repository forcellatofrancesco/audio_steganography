import tomllib
import csv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from collections import Counter

from audio.audiowaves import load_waveform_from_file
from util.decoding import decode_and_score_message


def load_config_file(config_path: Path) -> dict:
    """Load the TOML configuration file."""
    with config_path.open("rb") as config_file:
        return tomllib.load(config_file)


def report_progress(completed: int, total: int, last_printed: int) -> int:
    """Print progress when the integer percentage changes."""
    if total <= 0:
        return last_printed
    current = int(100 * completed / total)
    if current > last_printed:
        print(f"Progress: {current}% ({completed}/{total})")
        return current
    return last_printed


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
    config = load_config_file(Path("config.toml"))
    encoding_decoding_algorithm = config["algorithm"]["encoding_decoding"]
    preamble = config["sync"]["preamble"]
    cycles_per_symbol = config["audio"]["cycles_per_symbol"]

    input_path = config["output"]["automation_runs_csv"]
    output_path = config["output"]["decoded_csv"]

    with open(input_path, "r", newline="", encoding="utf-8") as csv_input, open(
        output_path, "w", newline="", encoding="utf-8"
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
                "Warning: duplicate download_path values found. Some rows likely reference overwritten files."
                f" Duplicated paths: {len(duplicated_paths)}"
            )

        total_rows = len(rows)
        last_printed = -1

        with ThreadPoolExecutor(max_workers=24) as executor:
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
                csv_out.flush()

                last_printed = report_progress(idx, total_rows, last_printed)


if __name__ == "__main__":
    main()
