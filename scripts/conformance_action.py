#!/usr/bin/env python3
"""Run one conformance executable and record unambiguous termination evidence."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys


def write_metadata(path: Path, exit_status: int, termination: str, signum: int) -> None:
    path.write_text(
        f"{exit_status}\t{termination}\t{signum}\n", encoding="utf-8"
    )


def run(args: argparse.Namespace) -> int:
    output_mode = "ab" if args.append else "wb"
    with args.output.open(output_mode) as output:
        try:
            process = subprocess.Popen(
                args.command,
                cwd=args.cwd,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except FileNotFoundError as error:
            output.write(f"{error}\n".encode())
            write_metadata(args.metadata, 127, "exec-error", 0)
            return 127
        except OSError as error:
            output.write(f"{error}\n".encode())
            write_metadata(args.metadata, 126, "exec-error", 0)
            return 126

        try:
            return_code = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timeout_signal = signal.SIGTERM
            os.killpg(process.pid, timeout_signal)
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                timeout_signal = signal.SIGKILL
                os.killpg(process.pid, timeout_signal)
                process.wait()
            write_metadata(args.metadata, 124, "timeout", timeout_signal)
            return 124

    if return_code < 0:
        signum = -return_code
        exit_status = 128 + signum
        write_metadata(args.metadata, exit_status, "signal", signum)
        return exit_status
    write_metadata(args.metadata, return_code, "exit", 0)
    return return_code


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--cwd", type=Path, required=True)
    result.add_argument("--timeout", type=float, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--metadata", type=Path, required=True)
    result.add_argument("--append", action="store_true")
    result.add_argument("command", nargs=argparse.REMAINDER)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser().error("missing command")
    if args.timeout <= 0:
        parser().error("timeout must be positive")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
