#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_REMOTE_ROOT = "/root/harnessgym_remote"
EXCLUDES = (
    ".git",
    ".pytest_cache",
    "__pycache__",
    ".harnessgym/runs",
    ".harnessgym_build",
    "*.pyc",
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync this example workspace to a CUDA host, then run a command there."
    )
    parser.add_argument("--host", default=os.environ.get("HARNESSGYM_GPU_HOST"))
    parser.add_argument("--port", default=os.environ.get("HARNESSGYM_GPU_PORT", "22"))
    parser.add_argument("--key", default=os.environ.get("HARNESSGYM_GPU_KEY"))
    parser.add_argument("--workspace", default=os.environ.get("HARNESSGYM_GPU_WORKSPACE", "triton_rmsnorm_h100"))
    parser.add_argument("--remote-root", default=os.environ.get("HARNESSGYM_GPU_REMOTE_ROOT", DEFAULT_REMOTE_ROOT))
    parser.add_argument("--no-clean", action="store_true", help="Do not remove the remote workspace before extracting.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Remote command after --, for example -- python3 verifier.py")
    args = parser.parse_args(argv)
    if not args.host:
        parser.error("provide --host or HARNESSGYM_GPU_HOST")
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("provide a remote command after --")
    return args


def ssh_base(args: argparse.Namespace) -> list[str]:
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-p",
        str(args.port),
    ]
    if args.key:
        command.extend(["-i", str(Path(args.key).expanduser())])
    command.append(args.host)
    return command


def remote_dir(args: argparse.Namespace) -> str:
    safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in args.workspace)
    return f"{str(args.remote_root).rstrip('/')}/{safe_name}"


def sync_workspace(args: argparse.Namespace) -> str:
    destination = remote_dir(args)
    clean_prefix = f"rm -rf {shlex.quote(destination)} && " if not args.no_clean else ""
    remote_extract = (
        f"{clean_prefix}mkdir -p {shlex.quote(destination)} && "
        f"tar -xzf - -C {shlex.quote(destination)}"
    )
    tar_command = ["tar", "-czf", "-"]
    for pattern in EXCLUDES:
        tar_command.append(f"--exclude={pattern}")
    tar_command.append(".")

    tar_proc = subprocess.Popen(tar_command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ssh_proc = subprocess.Popen(
        [*ssh_base(args), remote_extract],
        stdin=tar_proc.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
    )
    assert tar_proc.stdout is not None
    tar_proc.stdout.close()
    ssh_stdout, ssh_stderr = ssh_proc.communicate()
    tar_stderr = tar_proc.stderr.read() if tar_proc.stderr is not None else b""
    tar_return = tar_proc.wait()
    if tar_return != 0:
        sys.stderr.write(tar_stderr.decode("utf-8", errors="replace"))
        raise SystemExit(tar_return)
    if ssh_proc.returncode != 0:
        sys.stdout.buffer.write(ssh_stdout)
        sys.stderr.buffer.write(ssh_stderr)
        raise SystemExit(ssh_proc.returncode)
    return destination


def run_remote(args: argparse.Namespace, destination: str) -> int:
    remote_command = f"cd {shlex.quote(destination)} && {shlex.join(args.command)}"
    process = subprocess.run(
        [*ssh_base(args), remote_command],
        text=True,
        capture_output=True,
        check=False,
    )
    sys.stdout.write(process.stdout)
    sys.stderr.write(process.stderr)
    return int(process.returncode)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    destination = sync_workspace(args)
    return run_remote(args, destination)


if __name__ == "__main__":
    raise SystemExit(main())
