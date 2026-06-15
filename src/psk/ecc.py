import numpy as np

from bit_phase.bit_phase import bits_to_int, int_to_bit_list
from util.types import bit_array

ECC_SCHEME_NONE = 0
ECC_SCHEME_HAMMING_7_4 = 1

HEADER_VERSION = 1
HEADER_VERSION_BITS = 2
ECC_SCHEME_BITS = 2
PAYLOAD_LENGTH_BITS = 16
MAX_PAYLOAD_LENGTH_BYTES = (1 << PAYLOAD_LENGTH_BITS) - 1

HEADER_DATA_BITS = HEADER_VERSION_BITS + ECC_SCHEME_BITS + PAYLOAD_LENGTH_BITS
HEADER_CODEWORD_DATA_BITS = 4
HEADER_CODEWORD_BITS = 7
ENCODED_HEADER_BITS = (
    HEADER_DATA_BITS // HEADER_CODEWORD_DATA_BITS
) * HEADER_CODEWORD_BITS


def pad_bits(bits: bit_array, block_size: int) -> tuple[bit_array, int]:
    bits = np.asarray(bits, dtype=np.int8).reshape(-1)
    remainder = len(bits) % block_size
    if remainder == 0:
        return bits.copy(), 0
    padding = block_size - remainder
    return np.concatenate([bits, np.zeros(padding, dtype=np.int8)]), padding


def _hamming74_encode_nibble(nibble: bit_array) -> bit_array:
    nibble = np.asarray(nibble, dtype=np.int8).reshape(-1)
    if len(nibble) != 4:
        raise ValueError("Hamming(7,4) encoding requires 4 data bits.")
    d1, d2, d3, d4 = nibble
    p1 = d1 ^ d2 ^ d4
    p2 = d1 ^ d3 ^ d4
    p4 = d2 ^ d3 ^ d4
    return np.array([p1, p2, d1, p4, d2, d3, d4], dtype=np.int8)


def _hamming74_decode_codeword(codeword: bit_array) -> tuple[bit_array, bool]:
    codeword = np.asarray(codeword, dtype=np.int8).reshape(-1)
    if len(codeword) != 7:
        raise ValueError("Hamming(7,4) decoding requires 7 bits.")
    b1, b2, b3, b4, b5, b6, b7 = codeword
    s1 = b1 ^ b3 ^ b5 ^ b7
    s2 = b2 ^ b3 ^ b6 ^ b7
    s4 = b4 ^ b5 ^ b6 ^ b7
    syndrome = s1 + (s2 << 1) + (s4 << 2)
    corrected = False
    if syndrome:
        index = syndrome - 1
        if 0 <= index < 7:
            codeword[index] ^= 1
            corrected = True
    data_bits = np.array(
        [codeword[2], codeword[4], codeword[5], codeword[6]], dtype=np.int8
    )
    return data_bits, corrected


def encode_hamming_7_4(bits: bit_array) -> bit_array:
    bits = np.asarray(bits, dtype=np.int8).reshape(-1)
    if len(bits) % 4 != 0:
        raise ValueError("Hamming(7,4) encoding requires 4-bit blocks.")
    if bits.size == 0:
        return np.array([], dtype=np.int8)
    encoded = np.empty((bits.size // 4) * 7, dtype=np.int8)
    out_idx = 0
    for idx in range(0, len(bits), 4):
        encoded[out_idx : out_idx + 7] = _hamming74_encode_nibble(bits[idx : idx + 4])
        out_idx += 7
    return encoded


def decode_hamming_7_4(bits: bit_array) -> tuple[bit_array, int]:
    bits = np.asarray(bits, dtype=np.int8).reshape(-1)
    corrections = 0
    usable_len = len(bits) - (len(bits) % 7)
    if usable_len == 0:
        return np.array([], dtype=np.int8), 0

    decoded = np.empty((usable_len // 7) * 4, dtype=np.int8)
    out_idx = 0
    for idx in range(0, usable_len, 7):
        data_bits, corrected = _hamming74_decode_codeword(bits[idx : idx + 7])
        decoded[out_idx : out_idx + 4] = data_bits
        out_idx += 4
        corrections += 1 if corrected else 0
    return decoded, corrections


def build_header_bits(payload_length_bytes: int, ecc_scheme: int) -> bit_array:
    if payload_length_bytes < 0 or payload_length_bytes > MAX_PAYLOAD_LENGTH_BYTES:
        raise ValueError("payload length exceeds header limits.")
    header = np.empty(HEADER_DATA_BITS, dtype=np.int8)
    header[0:HEADER_VERSION_BITS] = int_to_bit_list(HEADER_VERSION, HEADER_VERSION_BITS)
    header[HEADER_VERSION_BITS : HEADER_VERSION_BITS + ECC_SCHEME_BITS] = (
        int_to_bit_list(ecc_scheme, ECC_SCHEME_BITS)
    )
    header[HEADER_VERSION_BITS + ECC_SCHEME_BITS : HEADER_DATA_BITS] = int_to_bit_list(
        payload_length_bytes, PAYLOAD_LENGTH_BITS
    )
    return header


def parse_header_bits(header_bits: bit_array) -> tuple[int, int, int]:
    header_bits = np.asarray(header_bits, dtype=np.int8).reshape(-1)
    if len(header_bits) < HEADER_DATA_BITS:
        raise ValueError("header bits are incomplete.")
    version = bits_to_int(header_bits[:HEADER_VERSION_BITS])
    ecc_scheme = bits_to_int(
        header_bits[HEADER_VERSION_BITS : HEADER_VERSION_BITS + ECC_SCHEME_BITS]
    )
    payload_length = bits_to_int(
        header_bits[HEADER_VERSION_BITS + ECC_SCHEME_BITS : HEADER_DATA_BITS]
    )
    return version, ecc_scheme, payload_length
