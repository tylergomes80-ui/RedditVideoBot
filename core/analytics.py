import sqlite3, time, json
from pathlib import Path

def db_path(root: str) -> str:
    p = Path(root)/"analytics.sqlite3"
    return str(p)

def ensure_schema(dbfile: str):
    con = sqlite3.connect(dbfile)
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS renders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts INTEGER,
        config TEXT,
        output TEXT,
        duration_ms INTEGER,
        notes TEXT
    )""")
    con.commit(); con.close()

def record(root: str, config: dict, output: str, duration_ms: int, notes:str=""):
    db = db_path(root); ensure_schema(db)
    con = sqlite3.connect(db); cur = con.cursor()
    cur.execute("INSERT INTO renders(ts,config,output,duration_ms,notes) VALUES (?,?,?,?,?)",
                (int(time.time()), json.dumps(config), output, duration_ms, notes))
    con.commit(); con.close()
