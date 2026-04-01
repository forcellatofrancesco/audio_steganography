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
from typing import Any

from playwright.sync_api import sync_playwright

from audio.audio_config import AudioConfig
from automation.config import WhatsAppConfig
from automation.whatsapp import WhatsAppAutomation
import itertools
import numpy as np

from main_encode import encode_from_config


def _normalize_param(value: Any) -> str:
    """Normalize numeric values to stable strings for resume matching."""
    try:
        return f"{float(value):.12g}"
    except (TypeError, ValueError):
        return str(value)


def _params_key(params: dict[str, Any], keys: list[str]) -> tuple[str, ...]:
    return tuple(_normalize_param(params[k]) for k in keys)


def main():
    config_file = None
    with open("config.toml", "rb") as f:
        config_file = tomllib.load(f)
    encoding_decoding_algorithm = config_file["algorithm"]["encoding_decoding"]
    message = None
    with open(config_file["input"]["message"], "r") as input:
        message = "".join(input.readlines())
    preamble: list[int] = config_file["sync"]["preamble"]
    byte_data = message.encode("utf-8")
    # Load WhatsApp configuration (edit config.toml to customize)
    config = WhatsAppConfig()
    config.profile_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        # Launch persistent browser context to maintain login session
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(config.profile_dir),
            headless=config.headless,
            args=["--start-maximized"],
            viewport={"width": 1270, "height": 720},
        )

        try:
            # Instantiate WhatsApp automation
            # SWITCH PLATFORMS HERE: Replace with other social media class
            automation = WhatsAppAutomation(context, config)
            automation.navigate_to_whatsapp()
            print("Waiting for WhatsApp Web. Scan QR code if prompted...")
            automation.load_main_ui()

            print("Now open the target chat manually.")
            print("Waiting for the recording button to appear...")

            # param_grid = {
            #     "frequency": np.arange(200, 15000, 100),
            #     "volume_gain_data": np.arange(0.15, 1.0, 0.05),
            #     "volume_noise": np.arange(0.01, 1.0, 0.05),
            # }
            param_grid = {
                "frequency": np.linspace(150, 350, num=10),
                "volume_gain_data": np.linspace(0.15, 1.0, num=10),
                "volume_noise": np.linspace(0.01, 1.0, num=10),
            }
            keys = list(param_grid.keys())
            values = list(param_grid.values())
            total = sum(1 for _ in itertools.product(*values))

            csv_output_path = Path(config_file["output"]["automation_runs_csv"])
            csv_output_path.parent.mkdir(parents=True, exist_ok=True)
            csv_exists = csv_output_path.exists()

            completed_keys: set[tuple[str, ...]] = set()
            if csv_exists:
                with csv_output_path.open(
                    "r", newline="", encoding="utf-8"
                ) as csv_file:
                    reader = csv.DictReader(csv_file)
                    for row in reader:
                        if row.get("status", "").strip().lower() != "success":
                            continue
                        if not all(k in row for k in keys):
                            continue
                        completed_keys.add(
                            tuple(_normalize_param(row[k]) for k in keys)
                        )

            print(
                f"Resume status: {len(completed_keys)}/{total} parameter sets already completed."
            )
            with csv_output_path.open("a", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=[*keys, "download_path", "status"],
                )
                if not csv_exists:
                    writer.writeheader()
                percentage = 0.0
                for combo in itertools.product(*values):
                    params = dict(zip(keys, combo))
                    current_key = _params_key(params, keys)
                    if current_key in completed_keys:
                        continue
                    p = len(completed_keys) / total * 100.0
                    if p > percentage + 1.0:
                        percentage = p
                        print(f"Completed: {p}%")
                    try:
                        audio_config = AudioConfig(
                            frequency=params["frequency"],
                            high_pass_filter=params["frequency"] + 50,
                            volume_gain_data=params["volume_gain_data"],
                            volume_noise=params["volume_noise"],
                        )
                        encode_from_config(
                            byte_data,
                            preamble,
                            encoding_decoding_algorithm,
                            config_file["dataset"]["directory"],
                            config_file["output"]["waveform"],
                            config_file["algorithm"].get("shuffle_seed"),
                            audio_config,
                        )
                        # Execute the complete workflow
                        downloaded_path = automation.run()

                        writer.writerow(
                            {
                                **params,
                                "download_path": (
                                    str(downloaded_path) if downloaded_path else ""
                                ),
                                "status": "success",
                            }
                        )
                        csv_file.flush()
                        completed_keys.add(current_key)
                    except Exception:
                        writer.writerow(
                            {
                                **params,
                                "download_path": "",
                                "status": "failed",
                            }
                        )
                        csv_file.flush()
                        raise

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
