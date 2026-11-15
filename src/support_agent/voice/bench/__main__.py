"""Command line: `python -m support_agent.voice.bench make-audio | stt | calls | report`."""

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="support_agent.voice.bench")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("make-audio", help="write the test audio to bench-audio/")
    stt = sub.add_parser("stt", help="measure the word error rate and time of a Whisper model")
    stt.add_argument("--model", required=True, help="MLX Whisper repository")
    stt.add_argument("--rate", type=int, choices=(16000, 8000), default=16000)
    calls = sub.add_parser("calls", help="run the scripted calls through the full pipeline")
    calls.add_argument("--rate", type=int, choices=(16000, 8000), default=16000)
    report = sub.add_parser("report", help="write a calls report again from its raw JSON")
    report.add_argument("json", help="the raw result JSON next to a calls report")
    args = parser.parse_args(argv)

    if args.command == "make-audio":
        from support_agent.voice.bench.make_audio import make_audio

        print(f"Wrote {make_audio()} audio files.")
    elif args.command == "stt":
        from support_agent.voice.bench.stt import run_stt, write_report

        print(f"Report: {write_report(run_stt(args.model, args.rate))}")
    elif args.command == "report":
        from support_agent.voice.bench.pipeline_bench import rebuild_report

        print(f"Report: {rebuild_report(args.json)}")
    else:
        import asyncio

        from support_agent.voice.bench.pipeline_bench import run_calls
        from support_agent.voice.bench.pipeline_bench import write_report as write_calls_report

        print(f"Report: {write_calls_report(asyncio.run(run_calls(args.rate)))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
