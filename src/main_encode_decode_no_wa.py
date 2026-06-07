"""Main orchestrator for social media voice message automation.

This module is the entry point for the voice message recording and transmission
automation framework. It handles browser setup and delegates to the appropriate
social media implementation.

To switch between social media platforms, change the import and instantiation
at the bottom of this file. Currently uses WhatsApp.
"""

import tomllib
import csv
from pathlib import Path
from typing import Any, Sequence
from concurrent.futures import ThreadPoolExecutor

from audio.audio_config import AudioConfig
import itertools
import numpy as np

from audio.audiowaves import load_waveform_from_file
from main_encode import encode_from_config
from util.decoding import decode_and_score_message
from util import test_messages


def _normalize_param(value: Any) -> str:
    """Normalize numeric values to stable strings for resume matching."""
    try:
        return f"{float(value):.12g}"
    except (TypeError, ValueError):
        return str(value)


def _params_key(params: dict[str, Any], keys: list[str]) -> tuple[str, ...]:
    return tuple(_normalize_param(params[k]) for k in keys)


def load_config_file(config_path: Path) -> dict[str, Any]:
    """Load the TOML configuration file."""
    with config_path.open("rb") as config_file:
        return tomllib.load(config_file)


def load_completed_parameter_keys(
    csv_output_path: Path,
    keys: list[str],
) -> set[tuple[str, ...]]:
    """Read previously completed successful parameter sets from the CSV output."""
    completed_keys: set[tuple[str, ...]] = set()
    if not csv_output_path.exists():
        return completed_keys

    with csv_output_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row.get("status", "").strip().lower() != "success":
                continue
            if not all(k in row for k in keys):
                continue
            completed_keys.add(tuple(_normalize_param(row[k]) for k in keys))

    return completed_keys


def build_pending_parameters(
    param_grid: dict[str, Any],
    completed_keys: set[tuple[str, ...]],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Return the parameter keys and the combinations that still need processing."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    pending_params: list[dict[str, Any]] = []

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        if _params_key(params, keys) in completed_keys:
            continue
        pending_params.append(params)

    return keys, pending_params


def decode_row(
    message: str,
    frequency: int,
    download_path: str,
    preamble: np.ndarray,
    cycles_per_symbol: float,
    encoding_decoding_algorithm: str,
):
    """Decode a single row's audio data."""
    waveform, sample_rate = load_waveform_from_file(download_path)
    return decode_and_score_message(
        waveform,
        sample_rate,
        message,
        preamble,
        frequency,
        cycles_per_symbol,
        encoding_decoding_algorithm,
    )


def process_parameter_set(
    index: int,
    params: dict[str, Any],
    preamble: np.ndarray,
    encoding_decoding_algorithm: str,
    config_file: dict[str, Any],
):
    """Encode and decode a single parameter combination."""
    try:
        audio_config = AudioConfig(
            frequency=params["frequency"],
            # I tried changing the value but it kept getting worse
            high_pass_filter=params["frequency"] + 50,
            volume_gain_data=params["volume_gain_data"],
        )
        byte_data = params["message"].encode("utf-8")
        temp_audio_file = f"output/temps/temp_{index:06d}.wav"
        Path(temp_audio_file).parent.mkdir(parents=True, exist_ok=True)
        encode_from_config(
            byte_data,
            preamble,
            encoding_decoding_algorithm,
            config_file["dataset"]["directory"],
            temp_audio_file,
            config_file["algorithm"].get("shuffle_seed"),
            audio_config,
        )

        decoded_row = decode_row(
            params["message"],
            params["frequency"],
            temp_audio_file,
            preamble,
            audio_config.cycles_per_symbol,
            encoding_decoding_algorithm,
        )
        return {
            **params,
            "volume_noise": audio_config.volume_noise,
            "error_rate": decoded_row["error_rate"],
            "decoded_data": decoded_row["decoded_data"],
            "status": "success",
            "decode_status": decoded_row["decode_status"],
            "exception": None,
        }
    except Exception as exc:
        return {
            **params,
            "status": "failed",
            "exception": exc,
        }


def result_row(result: dict[str, Any], fieldnames: Sequence[str]) -> dict[str, Any]:
    """Project a result into the CSV schema."""
    return {k: result.get(k, "") for k in fieldnames}


def report_progress(
    completed_total: int,
    total: int,
    last_printed_percentage: int,
) -> int:
    """Print progress when the integer percentage changes."""
    current_percentage = int(100 * completed_total / total)
    if current_percentage > last_printed_percentage:
        print(f"Completed: {current_percentage}% ({completed_total}/{total})")
        return current_percentage
    return last_printed_percentage


def main():
    config_file = load_config_file(Path("config.toml"))
    encoding_decoding_algorithm = config_file["algorithm"]["encoding_decoding"]
    preamble = np.asarray(config_file["sync"]["preamble"], dtype=np.int8)

    csv_output_path = Path(config_file["output"]["encode_decode_no_wa"])
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    # Parameter grid: set of experiments/tests that will be executed
    param_grid = {
        "frequency": np.linspace(150, 400, num=10).astype(int),
        "volume_gain_data": np.linspace(0.15, 1.0, num=10),
        "message": test_messages.messages,
    }
    keys = list(param_grid.keys())
    total = sum(1 for _ in itertools.product(*param_grid.values()))
    completed_keys = load_completed_parameter_keys(csv_output_path, keys)
    _, pending_params = build_pending_parameters(param_grid, completed_keys)

    print(
        f"Resume status: {len(completed_keys)}/{total} parameter sets already completed."
    )
    with csv_output_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                *keys,
                "volume_noise",
                "status",
                "error_rate",
                "decoded_data",
                "decode_status",
            ],
        )
        if csv_output_path.stat().st_size == 0:
            writer.writeheader()

        total_pending = len(pending_params)
        last_printed_percentage = int(len(completed_keys) / total * 100.0)

        with ThreadPoolExecutor(max_workers=24) as executor:
            for result in executor.map(
                process_parameter_set,
                range(total_pending),
                pending_params,
                itertools.repeat(preamble),
                itertools.repeat(encoding_decoding_algorithm),
                itertools.repeat(config_file),
            ):
                writer.writerow(result_row(result, list(writer.fieldnames or [])))
                csv_file.flush()

                if result["status"] == "success":
                    completed_keys.add(_params_key(result, keys))
                else:
                    raise RuntimeError(
                        "Processing failed for parameter set "
                        f"{_params_key(result, keys)}"
                    ) from result["exception"]

                completed_total = len(completed_keys)
                last_printed_percentage = report_progress(
                    completed_total,
                    total,
                    last_printed_percentage,
                )


if __name__ == "__main__":
    main()
