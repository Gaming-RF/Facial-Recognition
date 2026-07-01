"""
SQLite storage for persons and face encodings.
Compatible with the existing faces.db schema.
"""
import sqlite3
import json
import time
import numpy as np
from app.config import DB_PATH, DATA_DIR, MUGSHOTS_DIR


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create tables if they don't exist. Safe to call repeatedly."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MUGSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            created_at REAL NOT NULL,
            mugshot_path TEXT,
            attributes TEXT DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS encodings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL,
            encoding BLOB NOT NULL,
            created_at REAL NOT NULL,
            FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
        );
    """)
    # Migration: add attributes column if missing
    cols = [r[1] for r in conn.execute("PRAGMA table_info(persons)").fetchall()]
    if "attributes" not in cols:
        conn.execute("ALTER TABLE persons ADD COLUMN attributes TEXT DEFAULT '{}'")
    conn.commit()
    conn.close()


# ── Person CRUD ─────────────────────────────────────────────────

def add_person(name: str, encodings=None, mugshot_path=None):
    now = time.time()
    conn = get_db()
    try:
        cur = conn.execute(
            "INSERT INTO persons (name, created_at, mugshot_path) VALUES (?, ?, ?)",
            (name, now, str(mugshot_path) if mugshot_path else None),
        )
        person_id = cur.lastrowid
        if encodings:
            for enc in encodings:
                conn.execute(
                    "INSERT INTO encodings (person_id, encoding, created_at) VALUES (?, ?, ?)",
                    (person_id, json.dumps(enc.tolist()), now),
                )
        conn.commit()
        return person_id
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()


def get_all_persons():
    conn = get_db()
    rows = conn.execute("SELECT * FROM persons ORDER BY name COLLATE NOCASE").fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["attributes"] = json.loads(d.get("attributes") or "{}")
        d["encoding_count"] = get_encoding_count(d["id"])
        result.append(d)
    return result


def get_person_by_id(person_id: int):
    conn = get_db()
    row = conn.execute("SELECT * FROM persons WHERE id = ?", (person_id,)).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    d["attributes"] = json.loads(d.get("attributes") or "{}")
    d["encoding_count"] = get_encoding_count(d["id"])
    return d


def delete_person(person_id: int):
    conn = get_db()
    conn.execute("DELETE FROM persons WHERE id = ?", (person_id,))
    conn.commit()
    conn.close()


def update_person_attributes(person_id: int, attributes: dict):
    conn = get_db()
    conn.execute(
        "UPDATE persons SET attributes = ? WHERE id = ?",
        (json.dumps(attributes), person_id),
    )
    conn.commit()
    conn.close()


# ── Encoding CRUD ───────────────────────────────────────────────

def add_encoding(person_id: int, encoding: np.ndarray):
    now = time.time()
    conn = get_db()
    conn.execute(
        "INSERT INTO encodings (person_id, encoding, created_at) VALUES (?, ?, ?)",
        (person_id, json.dumps(encoding.tolist()), now),
    )
    conn.commit()
    conn.close()


def get_encoding_count(person_id: int) -> int:
    conn = get_db()
    count = conn.execute(
        "SELECT COUNT(*) FROM encodings WHERE person_id = ?", (person_id,)
    ).fetchone()[0]
    conn.close()
    return count


def get_all_encodings() -> list[dict]:
    conn = get_db()
    rows = conn.execute("""
        SELECT e.id, e.encoding, p.name, p.id AS person_id
        FROM encodings e JOIN persons p ON e.person_id = p.id
    """).fetchall()
    conn.close()
    result = []
    for r in rows:
        result.append({
            "id": r["id"],
            "encoding": np.array(json.loads(r["encoding"]), dtype=np.float32),
            "name": r["name"],
            "person_id": r["person_id"],
        })
    return result


def delete_encoding(enc_id: int):
    conn = get_db()
    conn.execute("DELETE FROM encodings WHERE id = ?", (enc_id,))
    conn.commit()
    conn.close()


def get_encodings_for_person(person_id: int) -> list[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT id, encoding, created_at FROM encodings WHERE person_id = ?",
        (person_id,),
    ).fetchall()
    conn.close()
    return [{"id": r["id"], "encoding": json.loads(r["encoding"]), "created_at": r["created_at"]} for r in rows]
