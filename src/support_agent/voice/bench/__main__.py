"""Command line: `python -m support_agent.voice.bench make-audio | stt`."""

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="support_agent.voice.bench")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("make-audio", help="write the test audio to bench-audio/")
    stt = sub.add_parser("stt", help="measure the word error rate and time of a Whisper model")
    stt.add_argument("--model", required=True, help="MLX Whisper repository")
    stt.add_argument("--rate", type=int, choices=(16000, 8000), default=16000)
    args = parser.parse_args(argv)

    if args.command == "make-audio":
        from support_agent.voice.bench.make_audio import make_audio

        print(f"Wrote {make_audio()} audio files.")
    else:
        from support_agent.voice.bench.stt import run_stt, write_report

        print(f"Report: {write_report(run_stt(args.model, args.rate))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
