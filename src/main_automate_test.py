"""Main orchestrator for social media voice message automation.

This module is the entry point for the voice message recording and transmission
automation framework. It handles browser setup and delegates to the appropriate
social media implementation.

To switch between social media platforms, change the import and instantiation
at the bottom of this file. Currently uses WhatsApp.
"""

import datetime
import tomllib
import csv
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright

from audio.audio_config import AudioConfig
from automation.config import WhatsAppConfig
from automation.whatsapp import WhatsAppAutomation
import itertools
import numpy as np

from main_encode import encode_from_config
from util import test_messages
from util.types import bit_array, value_array


def _normalize_param(value: Any) -> str:
    """Normalize numeric values to stable strings for resume matching."""
    try:
        return f"{float(value):.12g}"
    except (TypeError, ValueError):
        return str(value)


def _params_key(params: dict[str, Any], keys: list[str]) -> tuple[str, ...]:
    return tuple(_normalize_param(params[k]) for k in keys)


def _run_parameter_grid(
    automation: WhatsAppAutomation,
    config_file: dict[str, Any],
    preamble: bit_array,
    algorithm: str,
    csv_output_path: Path,
) -> None:
    """Run the full parameter grid for a single encoding algorithm."""
    print(f"Starting {algorithm} run. CSV output: {csv_output_path}")

    param_grid = {
        "frequency": np.linspace(200, 400, num=10).astype(int),
        "volume_gain_data": np.linspace(0.25, 0.9, num=5),
        "message": test_messages.messages,
        "cycles_per_symbol": [2, 5, 7],
    }
    # TODO: this is just for the synthetic tests
    # param_grid = {
    #     "frequency": list(itertools.repeat(250, 10)),
    #     "volume_gain_data": [0.7],
    #     "message": ["Hi!"],
    # }
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    total = sum(1 for _ in itertools.product(*values))

    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_exists = csv_output_path.exists()

    ######################################################### TODO: re-add it
    completed_keys: set[tuple[str, ...]] = set()
    if csv_exists:
        with csv_output_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                if row.get("status", "").strip().lower() != "success":
                    continue
                if not all(k in row for k in keys):
                    continue
                completed_keys.add(tuple(_normalize_param(row[k]) for k in keys))
    print(
        f"Resume status for {algorithm}: {len(completed_keys)}/{total} parameter sets already completed."
    )

    with csv_output_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[*keys, "volume_noise", "download_path", "type"],
        )
        if not csv_exists:
            writer.writeheader()
        percentage = 0.0
        for index, combo in enumerate(itertools.product(*values)):
            params = dict(zip(keys, combo))
            # TODO: READD IT
            current_key = _params_key(params, keys)
            if current_key in completed_keys:
                continue
            p = len(completed_keys) / total * 100.0
            if p > percentage + 1.0:
                percentage = p
                print(f"Completed {algorithm}: {p}%")
            try:
                audio_config = AudioConfig(
                    frequency=params["frequency"],
                    high_pass_filter=params["frequency"] + 50,
                    volume_gain_data=params["volume_gain_data"],
                    cycles_per_symbol=params["cycles_per_symbol"],
                )
                byte_data = params["message"].encode("utf-8")
                encode_from_config(
                    byte_data,
                    preamble,
                    algorithm,
                    config_file["dataset"]["directory"],
                    config_file["output"]["waveform"],
                    config_file["algorithm"].get("shuffle_seed"),
                    audio_config,
                )
                downloaded_path = automation.run()
                source = Path(
                    config_file["output"]["waveform"],
                )
                current_temp = f"{config_file['whatsapp_automation']['download_output_path']}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}{index:03}.wav"
                source.rename(current_temp)
                writer.writerow(
                    {
                        **params,
                        "volume_noise": audio_config.volume_noise,
                        "download_path": current_temp,
                        "type": "uncompressed",
                    }
                )
                writer.writerow(
                    {
                        **params,
                        "volume_noise": audio_config.volume_noise,
                        "download_path": (
                            str(downloaded_path) if downloaded_path else ""
                        ),
                        "type": "compressed",
                    }
                )
                csv_file.flush()

                # TODO: readd it
                completed_keys.add(current_key)
            except Exception:
                csv_file.flush()
                raise


def main():
    config_file = None
    with open("config.toml", "rb") as f:
        config_file = tomllib.load(f)
    preamble = np.asarray(config_file["sync"]["preamble"], dtype=np.int8)

    # Load WhatsApp configuration (edit config.toml to customize)
    config = WhatsAppConfig()
    config.profile_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        # Launch persistent browser context to maintain login session
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(config.profile_dir),
            headless=config.headless,
            args=[
                "--start-maximized",
                # Disable default browser audio processing
                "--goog-echo-cancellation=false",
                "--goog-auto-gain-control=false",
                "--goog-noise-suppression=false",
                "--goog-highpass-filter=false",
            ],
            viewport={"width": 1270, "height": 720},
        )
        # Disable auto-gain microphone features: stable virtual microphone
        context.add_init_script("""
            const origGUM = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
            navigator.mediaDevices.getUserMedia = function(constraints) {
                if (constraints && constraints.audio) {
                    if (typeof constraints.audio === 'boolean') {
                        constraints.audio = {};
                    }
                    constraints.audio.autoGainControl = false;
                    constraints.audio.noiseSuppression = false;
                    constraints.audio.echoCancellation = false;
                }
                return origGUM(constraints);
            };
        """)

        try:
            # Instantiate WhatsApp automation
            # SWITCH PLATFORMS HERE: Replace with other social media class
            automation = WhatsAppAutomation(context, config)
            automation.navigate_to_whatsapp()
            print("Waiting for WhatsApp Web. Scan QR code if prompted...")
            automation.load_main_ui()

            print("Now open the target chat manually once, then leave it open.")
            print("The script will run DBPSK first and then BPSK in the same chat.")

            automation_runs = [
                (
                    "bpsk",
                    Path("output/260618_ultimate_bpsk.csv"),
                ),
                (
                    "dbpsk",
                    Path("output/260618_ultimate_dbpsk.csv"),
                ),
            ]

            for algorithm, csv_output_path in automation_runs:
                _run_parameter_grid(
                    automation,
                    config_file,
                    preamble,
                    algorithm,
                    csv_output_path,
                )

            # Handle post-completion behavior
            if config.keep_open:
                print(
                    "Recording completed. Browser will stay open; press Ctrl+C to stop."
                )
                try:
                    while True:
                        context.pages[0].wait_for_timeout(1_000)
                except KeyboardInterrupt:
                    print("Stopping automation.")
            else:
                print("Recording completed. Closing browser in 3 seconds.")
                context.pages[0].wait_for_timeout(3_000)

        finally:
            context.close()


if __name__ == "__main__":
    main()
