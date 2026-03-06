from __future__ import annotations

from pathlib import Path


class VoiceManager:
    """Singleton manager for LibriSpeech FLAC metadata.

    The manager scans the LibriSpeech directory recursively, reads every
    ``.flac`` file, and builds a dictionary of:

    ``{file_path: duration_in_seconds}``
    """

    _instance: "VoiceManager | None" = None

    def __new__(cls, *args, **kwargs) -> "VoiceManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, librispeech_root: str) -> None:
        if self._initialized:
            return

        self._root = Path(librispeech_root)
        self._audio_durations: dict[str, float] = {}
        self._initialized = True

    @classmethod
    def get_instance(cls, librispeech_root: str) -> "VoiceManager":
        return cls(librispeech_root=librispeech_root)

    def build_audio_duration_index(
        self,
        force_refresh: bool = False,
    ) -> dict[str, float]:
        """Build and cache ``{file_path: duration}`` from LibriSpeech FLAC files."""
        if self._audio_durations and not force_refresh:
            return dict(self._audio_durations)

        if not self._root.exists():
            raise FileNotFoundError(f"LibriSpeech path not found: {self._root}")

        durations: dict[str, float] = {}
        for flac_path in sorted(self._root.rglob("*.flac")):
            audio_path = str(flac_path)
            durations[audio_path] = self._read_flac_duration_seconds(flac_path)

        self._audio_durations = durations
        return dict(self._audio_durations)

    def get_audio_duration_index(self) -> dict[str, float]:
        """Return a copy of the currently cached duration index."""
        return dict(self._audio_durations)

    def get_duration(self, audio_path: str) -> float | None:
        """Return duration in seconds for a given audio path, if available."""
        return self._audio_durations.get(audio_path)

    @staticmethod
    def _read_flac_duration_seconds(file_path: Path) -> float:
        """Read FLAC STREAMINFO metadata and compute duration.

        This avoids third-party dependencies by parsing FLAC metadata directly.
        """
        with file_path.open("rb") as stream:
            marker = stream.read(4)
            if marker != b"fLaC":
                raise ValueError(f"Invalid FLAC stream marker in file: {file_path}")

            streaminfo: bytes | None = None

            while True:
                header = stream.read(4)
                if len(header) != 4:
                    break

                is_last_block = bool(header[0] & 0x80)
                block_type = header[0] & 0x7F
                block_size = int.from_bytes(header[1:4], byteorder="big")

                block_data = stream.read(block_size)
                if len(block_data) != block_size:
                    raise ValueError(
                        f"Unexpected FLAC metadata EOF in file: {file_path}"
                    )

                if block_type == 0:
                    streaminfo = block_data
                    break

                if is_last_block:
                    break

        if streaminfo is None or len(streaminfo) < 18:
            raise ValueError(f"Missing FLAC STREAMINFO metadata in file: {file_path}")

        # Bytes 10..17 pack sample_rate/channels/bits_per_sample/total_samples.
        packed = int.from_bytes(streaminfo[10:18], byteorder="big")
        sample_rate = packed >> 44
        total_samples = packed & ((1 << 36) - 1)

        if sample_rate <= 0:
            raise ValueError(f"Invalid FLAC sample_rate in file: {file_path}")

        return total_samples / sample_rate


def get_voice_manager(librispeech_root: str = "input/LibriSpeech") -> VoiceManager:
    """Convenience accessor for singleton VoiceManager."""
    return VoiceManager.get_instance(librispeech_root=librispeech_root)
