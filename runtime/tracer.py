"""Append-only per-run evidence; no credentials in event payloads."""
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import time


def json_value(value):
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f'Unsupported trace type: {type(value).__name__}')


class Tracer:
    def __init__(self, path, run_id):
        self.path, self.run_id = Path(path), run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, event, **data):
        row = {'event': event, 'run_id': self.run_id, 'time_ns': time.time_ns(), **data}
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, default=json_value) + '\n')

    def read(self):
        return [json.loads(line) for line in self.path.read_text(encoding='utf-8').splitlines()]
