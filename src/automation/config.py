"""Configuration management for social media automation.

Handles reading and parsing configuration from config.toml.
Provides utilities for type conversion and default value handling.
"""

import tomllib
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent.parent


def _read_whatsapp_automation_config() -> dict:
    """Read WhatsApp automation configuration from config.toml."""
    config_path = BASE_DIR / "config.toml"
    with config_path.open("rb") as f:
        config = tomllib.load(f)
    section = config.get("whatsapp_automation", {})
    if not isinstance(section, dict):
        return {}
    return section


def _as_bool(value: Any, default: bool) -> bool:
    """Convert a value to boolean, with a default fallback."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _as_int(value: Any, default: int) -> int:
    """Convert a value to integer, with a default fallback."""
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default


def _as_str(value: Any, default: str) -> str:
    """Convert a value to string, with a default fallback."""
    if isinstance(value, str):
        return value.strip()
    return default


class AutomationConfig:
    """Base configuration class for social media automation.

    Can be extended by specific platform implementations.
    """

    def __init__(self, headless: bool = False, keep_open: bool = True):
        """Initialize base automation config.

        Args:
            headless: Run browser in headless mode
            keep_open: Keep browser open after completion
        """
        self.headless = headless
        self.keep_open = keep_open
        self.default_timeout_ms = 15_000


class WhatsAppConfig(AutomationConfig):
    """WhatsApp-specific automation configuration."""

    def __init__(self):
        """Initialize WhatsApp config from config.toml."""
        config_dict = _read_whatsapp_automation_config()

        super().__init__(
            headless=_as_bool(config_dict.get("headless"), False),
            keep_open=_as_bool(config_dict.get("keep_open"), True),
        )

        self.url = (
            _as_str(config_dict.get("url"), "https://web.whatsapp.com/")
            or "https://web.whatsapp.com/"
        )
        self.origin = (
            _as_str(config_dict.get("origin"), "https://web.whatsapp.com")
            or "https://web.whatsapp.com"
        )

        profile_dir_value = _as_str(
            config_dict.get("profile_dir"), ".playwright_whatsapp_profile"
        )
        self.profile_dir = Path(profile_dir_value).expanduser()
        if not self.profile_dir.is_absolute():
            self.profile_dir = (BASE_DIR / self.profile_dir).resolve()

        self.default_timeout_ms = _as_int(config_dict.get("default_timeout_ms"), 15_000)
        self.record_button_wait_timeout_ms = _as_int(
            config_dict.get("record_button_wait_timeout_ms"), 300_000
        )
        self.send_button_wait_timeout_ms = _as_int(
            config_dict.get("send_button_wait_timeout_ms"), 20_000
        )
        self.post_send_settle_delay_ms = _as_int(
            config_dict.get("post_send_settle_delay_ms"), 2_000
        )
        self.download_wait_timeout_ms = _as_int(
            config_dict.get("download_wait_timeout_ms"), 30_000
        )

        playback_script_value = _as_str(
            config_dict.get("playback_script_path"),
            "src/util/scripts/virtual_microphone/play_mic.sh",
        )
        self.playback_script_path = Path(playback_script_value).expanduser()
        if not self.playback_script_path.is_absolute():
            self.playback_script_path = (BASE_DIR / self.playback_script_path).resolve()

        download_output_value = _as_str(
            config_dict.get("download_output_path"), "output"
        )
        self.download_output_path = Path(download_output_value).expanduser()
        if not self.download_output_path.is_absolute():
            self.download_output_path = (BASE_DIR / self.download_output_path).resolve()
