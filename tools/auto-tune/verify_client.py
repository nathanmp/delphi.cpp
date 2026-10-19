#!/usr/bin/env python3
"""Verify the loaded context floor, then run the recipe's normal measurement client."""
from __future__ import annotations

import argparse
import json
import subprocess
import urllib.request
from pathlib import Path


def verify(props, floor):
    context = props.get('default_generation_settings', {}).get('n_ctx')
    if not isinstance(context, int) or context < floor:
        raise ValueError(f'context floor {floor} not verified: {context}')
    return {'verified_context': context}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--floor', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('client', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    with urllib.request.urlopen(args.url.rstrip('/') + '/props', timeout=30) as response:
        result = verify(json.load(response), args.floor)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    command = args.client[1:] if args.client[:1] == ['--'] else args.client
    if not command:
        parser.error('a measurement client command is required after --')
    return subprocess.run(command).returncode


if __name__ == '__main__':
    raise SystemExit(main())
