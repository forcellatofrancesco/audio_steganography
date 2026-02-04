# audio_encoding
This project is a POC that aims to prove the feasibility of encoding a message in the form of audio using Phase Shift Keying.

## Installation
Create a Python virtual environment called `.venv` (as the `encode_decode.sh` script uses this name).
Install all the packages in `requirements.txt`.

## Configuration
The configuration can be found in the `config.toml` file.
Make sure to create `input` and `output` folders.
Inside of input, make sure to add the message you want to encode.

## Run
To test encoding and decoding, simply run `encode_decode.sh`.