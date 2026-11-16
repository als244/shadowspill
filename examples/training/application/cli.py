"""Common connection and saved-settings handling for lightweight applications."""

import argparse
from dataclasses import fields, replace
from pathlib import Path

from shadowspill_client import SSH, ServiceClient

from .setup import read_resume


def argument_parser(description):
    parser = argparse.ArgumentParser(description=description)
    connection = parser.add_mutually_exclusive_group()
    connection.add_argument("--server", help="Existing HTTP(S) service URL")
    connection.add_argument("--ssh", help="SSH host alias of an existing service")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, help="Saved checkpoint reference JSON")
    return parser


def settings_from_args(args, settings_type, *, create=None, application=None):
    requested = {
        field.name: vars(args)[field.name]
        for field in fields(settings_type)
        if field.name in vars(args)
    }
    if args.resume is None:
        return (create or settings_type)(**requested), None
    resume, settings = read_resume(args.resume, settings_type)
    if application is not None and settings.application != application:
        raise ValueError("resume record belongs to another application")
    if replace(settings, **requested) != settings:
        raise ValueError("resume requires the saved training settings and schedule")
    return settings, resume


def connect(args):
    if args.ssh:
        return ServiceClient(ssh=SSH(args.ssh, args.server_port))
    return ServiceClient(base_url=args.server or f"http://127.0.0.1:{args.server_port}")


def run_directory(args):
    directory = args.run_dir.resolve()
    directory.mkdir(parents=True)  # Preserve all prior local runs.
    return directory
