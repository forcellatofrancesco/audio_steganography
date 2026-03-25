"""Audio configuration model and factory.

Supports creating typed audio config either from config.toml
or from configuration data provided programmatically.
"""

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any, Mapping


@dataclass(frozen=True)
class AudioConfig:
    """Typed configuration values used by the audio pipeline."""

    sample_rate: int = 48000
    frequency: int = 200
    high_pass_filter: float = 250.0
    cycles_per_symbol: float = 2.0
    volume_gain_data: float = 0.15
    volume_gain_voice: float = 1.0
    volume_noise: float = 0.01


class AudioConfigFactory:
    """Factory for creating AudioConfig instances.

    Args:
        config_path: Path to TOML file used when data is not provided.
        data: Optional configuration mapping. Can be either:
              - full config with an "audio" section
              - direct audio section mapping
    """

    def __init__(
        self,
        config_path: str | Path = "config.toml",
        data: Mapping[str, Any] | None = None,
    ):
        self._config_path = Path(config_path)
        self._data = data

    def create(self) -> AudioConfig:
        """Build an AudioConfig from constructor data or TOML file."""
        section = self._resolve_audio_section()

        config = AudioConfig(
            sample_rate=self._as_int(
                section.get("sample_rate"), AudioConfig.sample_rate
            ),
            frequency=self._as_int(section.get("frequency"), AudioConfig.frequency),
            high_pass_filter=self._as_float(
                section.get("high_pass_filter"), AudioConfig.high_pass_filter
            ),
            cycles_per_symbol=self._as_float(
                section.get("cycles_per_symbol"), AudioConfig.cycles_per_symbol
            ),
            volume_gain_data=self._as_float(
                section.get("volume_gain_data"), AudioConfig.volume_gain_data
            ),
            volume_gain_voice=self._as_float(
                section.get("volume_gain_voice"), AudioConfig.volume_gain_voice
            ),
            volume_noise=self._as_float(
                section.get("volume_noise"), AudioConfig.volume_noise
            ),
        )

        self._validate(config)
        return config

    def _resolve_audio_section(self) -> Mapping[str, Any]:
        if self._data is not None:
            if isinstance(self._data.get("audio"), Mapping):
                return self._data["audio"]
            return self._data

        with self._config_path.open("rb") as file:
            config = tomllib.load(file)

        section = config.get("audio", {})
        if isinstance(section, Mapping):
            return section
        return {}

    @staticmethod
    def _as_int(value: Any, default: int) -> int:
        if isinstance(value, bool):
            return default
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError:
                return default
        return default

    @staticmethod
    def _as_float(value: Any, default: float) -> float:
        if isinstance(value, bool):
            return default
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return default
        return default

    @staticmethod
    def _validate(config: AudioConfig) -> None:
        if config.sample_rate <= 0:
            raise ValueError("audio.sample_rate must be positive")
        if config.frequency <= 0:
            raise ValueError("audio.frequency must be positive")
        if config.high_pass_filter <= 0:
            raise ValueError("audio.high_pass_filter must be positive")
        if config.cycles_per_symbol <= 0:
            raise ValueError("audio.cycles_per_symbol must be positive")
        if config.volume_gain_data < 0:
            raise ValueError("audio.volume_gain_data must be >= 0")
        if config.volume_gain_voice < 0:
            raise ValueError("audio.volume_gain_voice must be >= 0")
        if config.volume_noise < 0:
            raise ValueError("audio.volume_noise must be >= 0")
