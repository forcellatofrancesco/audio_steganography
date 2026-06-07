import tomllib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from collections import Counter

import pandas as pd

from audio.audiowaves import load_waveform_from_file
from audio.audio_config import AudioConfigFactory
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

    audio_config = AudioConfigFactory().create()
    cycles_per_symbol = audio_config.cycles_per_symbol

    input_path = config["output"]["automation_runs_csv"]
    output_path = config["output"]["decoded_csv"]

    df = pd.read_csv(input_path)
    # df = df[(df["frequency"] > 200)]
    # print("_________ REMOVE FILTER FOR PROD ____________")
    rows = df.fillna("").to_dict(orient="records")
    print(len(rows))
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
        decoded_rows = []
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
            decoded_rows.append(result)
            last_printed = report_progress(idx, total_rows, last_printed)

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
    pd.DataFrame(decoded_rows).reindex(columns=keys).to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
