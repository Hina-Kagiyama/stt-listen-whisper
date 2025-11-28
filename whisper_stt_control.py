#!/usr/bin/env python3
import sys
import socket
import argparse

SOCKET_PATH = "/tmp/whisper_stt_socket"


def send_command(cmd: str) -> int:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.connect(SOCKET_PATH)
    except FileNotFoundError:
        print("Error: Whisper STT GUI is not running (socket not found).", file=sys.stderr)
        return 1
    except ConnectionRefusedError:
        print("Error: Cannot connect to Whisper STT GUI (connection refused).", file=sys.stderr)
        return 1

    try:
        s.sendall(cmd.encode("utf-8"))
    finally:
        s.close()
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Control Whisper STT GUI (start/stop/toggle)."
    )
    parser.add_argument(
        "command",
        choices=["start", "stop", "toggle"],
        help="Command to send to the running GUI.",
    )
    args = parser.parse_args()

    rc = send_command(args.command)
    sys.exit(rc)


if __name__ == "__main__":
    main()

