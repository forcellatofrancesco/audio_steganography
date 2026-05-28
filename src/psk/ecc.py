from bit_phase.bit_phase import bits_to_int, int_to_bit_list

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
ENCODED_HEADER_BITS = (HEADER_DATA_BITS // HEADER_CODEWORD_DATA_BITS) * HEADER_CODEWORD_BITS


def pad_bits(bits: list[int], block_size: int) -> tuple[list[int], int]:
    remainder = len(bits) % block_size
    if remainder == 0:
        return list(bits), 0
    padding = block_size - remainder
    return list(bits) + [0] * padding, padding


def _hamming74_encode_nibble(nibble: list[int]) -> list[int]:
    if len(nibble) != 4:
        raise ValueError("Hamming(7,4) encoding requires 4 data bits.")
    d1, d2, d3, d4 = nibble
    p1 = d1 ^ d2 ^ d4
    p2 = d1 ^ d3 ^ d4
    p4 = d2 ^ d3 ^ d4
    return [p1, p2, d1, p4, d2, d3, d4]


def _hamming74_decode_codeword(codeword: list[int]) -> tuple[list[int], bool]:
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
            codeword = list(codeword)
            codeword[index] ^= 1
            corrected = True
    data_bits = [codeword[2], codeword[4], codeword[5], codeword[6]]
    return data_bits, corrected


def encode_hamming_7_4(bits: list[int]) -> list[int]:
    if len(bits) % 4 != 0:
        raise ValueError("Hamming(7,4) encoding requires 4-bit blocks.")
    encoded: list[int] = []
    for idx in range(0, len(bits), 4):
        encoded.extend(_hamming74_encode_nibble(bits[idx : idx + 4]))
    return encoded


def decode_hamming_7_4(bits: list[int]) -> tuple[list[int], int]:
    decoded: list[int] = []
    corrections = 0
    usable_len = len(bits) - (len(bits) % 7)
    for idx in range(0, usable_len, 7):
        data_bits, corrected = _hamming74_decode_codeword(bits[idx : idx + 7])
        decoded.extend(data_bits)
        corrections += 1 if corrected else 0
    return decoded, corrections


def build_header_bits(payload_length_bytes: int, ecc_scheme: int) -> list[int]:
    if payload_length_bytes < 0 or payload_length_bytes > MAX_PAYLOAD_LENGTH_BYTES:
        raise ValueError("payload length exceeds header limits.")
    return (
        int_to_bit_list(HEADER_VERSION, HEADER_VERSION_BITS)
        + int_to_bit_list(ecc_scheme, ECC_SCHEME_BITS)
        + int_to_bit_list(payload_length_bytes, PAYLOAD_LENGTH_BITS)
    )


def parse_header_bits(header_bits: list[int]) -> tuple[int, int, int]:
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
