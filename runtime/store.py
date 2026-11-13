"""Single-machine run index; full traces and messages are separate artifacts."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3


class RunStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute('''CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY, status TEXT NOT NULL, reward REAL,
                trace_path TEXT NOT NULL, detail_json TEXT NOT NULL)''')

    def save(self, row):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute('INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?)',
                (row['run_id'], row['status'], row.get('reward'), row['trace_path'],
                 json.dumps(row, ensure_ascii=False)))

    def get(self, run_id):
        with closing(sqlite3.connect(self.path)) as db:
            row = db.execute('SELECT detail_json FROM runs WHERE run_id=?', (run_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def all(self):
        with closing(sqlite3.connect(self.path)) as db:
            rows = db.execute('SELECT detail_json FROM runs ORDER BY run_id').fetchall()
        return [json.loads(row[0]) for row in rows]

    def reserve(self, key, request, row):
        """Return (run_id, created, conflict); key and queued row commit together."""
        payload=json.dumps(request,sort_keys=True)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute('CREATE TABLE IF NOT EXISTS request_keys (key TEXT PRIMARY KEY, payload TEXT, run_id TEXT)')
            created=db.execute('INSERT OR IGNORE INTO request_keys VALUES (?,?,?)',
                               (key,payload,row['run_id'])).rowcount==1
            existing=db.execute('SELECT payload,run_id FROM request_keys WHERE key=?',(key,)).fetchone()
            if existing[0]!=payload:return existing[1],False,True
            if created:
                db.execute('INSERT INTO runs VALUES (?,?,?,?,?)',(row['run_id'],row['status'],None,
                    row['trace_path'],json.dumps(row,ensure_ascii=False)))
        return existing[1],created,False
