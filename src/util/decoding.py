"""Shared helpers for decoding and scoring recovered messages."""

from typing import Any

import numpy as np

from psk.psk_decoder import decode_from_audio
from util.types import bit_array, value_array


def decode_and_score_message(
    waveform: value_array,
    sample_rate: int,
    message: str,
    preamble: bit_array,
    frequency: int,
    cycles_per_symbol: float,
    algorithm: str,
) -> dict[str, Any]:
    """Decode a waveform and compare the recovered message with the expected text."""
    recovered_data, decode_status = decode_from_audio(
        waveform,
        preamble,
        sample_rate=sample_rate,
        frequency=frequency,
        cycles_per_symbol=cycles_per_symbol,
        algorithm=algorithm,
    )
    recovered_data = recovered_data.decode("utf-8", errors="replace")
    min_len = min(len(recovered_data), len(message))
    trimmed_data = recovered_data[:min_len]
    trimmed_message = message[:min_len]

    difference = sum(1 for a, b in zip(trimmed_data, trimmed_message) if a != b) + (
        len(message) - min_len
    )
    error_rate = difference / len(message)
    return {
        "error_rate": error_rate,
        "decoded_data": recovered_data,
        "decode_status": decode_status,
    }
