import argparse
import os
from pathlib import Path
import secrets


def write_epoch(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epoch = secrets.randbelow(0xFFFFFFFF) + 1
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(f'{epoch}\n', encoding='utf-8')
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    return epoch


def main():
    parser = argparse.ArgumentParser(
        description='Create the source_epoch shared by NX observation and time sync nodes.'
    )
    parser.add_argument(
        'path', nargs='?', default='/run/volleyball/nx_source_epoch'
    )
    args = parser.parse_args()
    epoch = write_epoch(args.path)
    print(epoch)
