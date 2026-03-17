# audio_stenography
This project is a POC that aims to prove the feasibility of encoding a message in the form of audio using Phase Shift Keying.

## Installation
Create a Python virtual environment called `.venv` (as the `encode_decode.sh` script uses this name).
Install all the packages in `requirements.txt`.
If you use non-WAV audio files (for example `.ogg`), install `ffmpeg` so the project can convert files to a waveform-compatible format automatically.

## Configuration
The configuration can be found in the `config.toml` file.
Make sure to create `input` and `output` folders.
Inside of input, make sure to add the message you want to encode.

## Run
To test encoding and decoding, simply run `encode_decode.sh`.

##
In order to run `main_automate_test.py` you need to run:
```sh
playwright install chromium
```
So that Playwright installs the dependency needed to automate the browser automation that allows to automatically send audios on WhatsApp Web.

Also configure WhatsApp automation values through `.env` in the project root:
```sh
cp .env.example .env
```
Then run the automation, manually open the target chat in WhatsApp Web, and it will:
1. click start recording,
2. run the playback script configured in `WHATSAPP_PLAYBACK_SCRIPT` (default: `src/util/scripts/virtual_microphone/play_mic.sh`),
3. click send (`Invia`) to send the recorded audio.