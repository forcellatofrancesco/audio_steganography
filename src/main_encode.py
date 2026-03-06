from psk.psk_encoder import (
    encode_to_audio,
)
import tomllib

from psk.utils import save_waveform_to_file
from voice_manager import get_voice_manager


def main():
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)
        encoding_decoding_algorithm = config["algorithm"]["encoding_decoding"]
        with open(config["input"]["message"], "r") as input:
            message = "".join(input.readlines())
            preamble: list[int] = config["sync"]["preamble"]
            byte_data = message.encode("utf-8")
            waveform = encode_to_audio(
                byte_data,
                preamble,
                sample_rate=config["audio"]["sample_rate"],
                frequency=config["audio"]["frequency"],
                cycles_per_symbol=config["audio"]["cycles_per_symbol"],
                algorithm=encoding_decoding_algorithm,
            )
            save_waveform_to_file(
                waveform,
                config["output"]["waveform"],
                sample_rate=config["audio"]["sample_rate"],
                volume=config["audio"]["volume"],
            )
            print(f"Audio file created: {config['output']['waveform']}")
            manager = get_voice_manager(config["dataset"]["directory"])
            audio_index = manager.build_audio_duration_index()

            # print(
            #     list(
            #         map(
            #             lambda x: x[0],
            #             sorted(
            #                 audio_index.items(),
            #                 key=lambda x: x[1],
            #             ),
            #         )
            #     )
            # )
            # print(audio_index.items())

            def get_audios(target_duration: float) -> list[tuple[str, float]]:
                possible = sorted(
                    filter(
                        lambda x: x[1] < target_duration,
                        audio_index.items(),
                    ),
                    key=lambda x: x[1],
                    reverse=True,
                )
                res = []
                res.append(possible[-1])
                total_duration = res[0][1]
                for path, duration in possible[:-1]:
                    if total_duration >= target_duration:
                        break
                    res.append((path, duration))
                    total_duration += duration
                return res

            print(get_audios(700))


if __name__ == "__main__":
    main()
