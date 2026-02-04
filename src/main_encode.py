from psk.psk_encoder import phase_shift_keying, save_waveform_to_file


def main():
    output_file_name = "output.wav"
    input_string = "hello wooooorld"
    byte_data = input_string.encode("utf-8")
    waveform = phase_shift_keying(byte_data)
    save_waveform_to_file(waveform, output_file_name)
    print("Audio file created: output.wav")


if __name__ == "__main__":
    main()
