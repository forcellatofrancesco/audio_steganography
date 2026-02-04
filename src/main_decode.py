from psk.psk_decoder import decode_phase_shift_keying, load_waveform_from_file


def main():
    output_file_name = "output.wav"
    waveform, sample_rate = load_waveform_from_file(output_file_name)
    recovered_data = decode_phase_shift_keying(waveform).decode("utf-8")
    print(f"Sample rate: {sample_rate}")
    print(f"Recovered message: '{recovered_data}'")


if __name__ == "__main__":
    main()
