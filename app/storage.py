import sqlite3
import json
import time
import config

_encoding_cache = None
_encoding_cache_time = 0.0
ENCODING_CACHE_TTL = 5.0


def get_db():
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.MUGSHOTS_DIR.mkdir(parents=True, exist_ok=True)
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
    # Migration: add attributes column if missing (for existing DBs)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(persons)").fetchall()]
    if "attributes" not in cols:
        conn.execute("ALTER TABLE persons ADD COLUMN attributes TEXT DEFAULT '{}'" )
        conn.commit()
    conn.commit()
    conn.close()


def add_person(name, encodings, mugshot_path=None):
    global _encoding_cache
    now = time.time()
    conn = get_db()
    try:
        cur = conn.execute(
            "INSERT INTO persons (name, created_at, mugshot_path) VALUES (?, ?, ?)",
            (name, now, str(mugshot_path) if mugshot_path else None)
        )
        person_id = cur.lastrowid
        for enc in encodings:
            conn.execute(
                "INSERT INTO encodings (person_id, encoding, created_at) VALUES (?, ?, ?)",
                (person_id, json.dumps(enc.tolist()), now)
            )
        conn.commit()
        _encoding_cache = None
        return person_id
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()


def get_all_persons():
    conn = get_db()
    rows = conn.execute("SELECT * FROM persons ORDER BY created_at DESC").fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["attributes"] = json.loads(d.get("attributes") or "{}")
        result.append(d)
    return result


def get_person_by_name(name):
    conn = get_db()
    row = conn.execute("SELECT * FROM persons WHERE name = ?", (name,)).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    d["attributes"] = json.loads(d.get("attributes") or "{}")
    return d


def get_all_encodings():
    global _encoding_cache, _encoding_cache_time
    now = time.time()
    if _encoding_cache is not None and (now - _encoding_cache_time) < ENCODING_CACHE_TTL:
        return _encoding_cache
    conn = get_db()
    rows = conn.execute("""
        SELECT e.encoding, p.name, p.id as person_id
        FROM encodings e JOIN persons p ON e.person_id = p.id
    """).fetchall()
    conn.close()
    result = []
    for r in rows:
        result.append({
            "encoding": json.loads(r["encoding"]),
            "name": r["name"],
            "person_id": r["person_id"]
        })
    _encoding_cache = result
    _encoding_cache_time = now
    return result


def add_encoding_to_person(person_id, encoding):
    """Add a new encoding to a person. Returns the new encoding count."""
    global _encoding_cache
    now = time.time()
    conn = get_db()
    conn.execute(
        "INSERT INTO encodings (person_id, encoding, created_at) VALUES (?, ?, ?)",
        (person_id, json.dumps(encoding.tolist()), now)
    )
    count = conn.execute(
        "SELECT COUNT(*) FROM encodings WHERE person_id = ?", (person_id,)
    ).fetchone()[0]
    conn.commit()
    conn.close()
    _encoding_cache = None
    return count


def get_encoding_count(person_id):
    conn = get_db()
    count = conn.execute(
        "SELECT COUNT(*) FROM encodings WHERE person_id = ?", (person_id,)
    ).fetchone()[0]
    conn.close()
    return count


def get_person_by_id(person_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM persons WHERE id = ?", (person_id,)).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    d["attributes"] = json.loads(d.get("attributes") or "{}")
    return d


def update_person_attributes(person_id, attributes):
    """Merge attributes into the person's attributes JSON."""
    conn = get_db()
    row = conn.execute("SELECT attributes FROM persons WHERE id = ?", (person_id,)).fetchone()
    if not row:
        conn.close()
        return False
    existing = json.loads(row[0] or "{}")
    existing.update(attributes)
    conn.execute("UPDATE persons SET attributes = ? WHERE id = ?",
                 (json.dumps(existing), person_id))
    conn.commit()
    conn.close()
    return True


def delete_person(person_id):
    global _encoding_cache
    conn = get_db()
    conn.execute("DELETE FROM persons WHERE id = ?", (person_id,))
    conn.commit()
    conn.close()
    _encoding_cache = None


def get_encodings_for_person(person_id):
    """Return all encoding IDs and their creation times for a person."""
    conn = get_db()
    rows = conn.execute(
        "SELECT id, created_at FROM encodings WHERE person_id = ? ORDER BY created_at",
        (person_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_encoding(encoding_id):
    global _encoding_cache
    conn = get_db()
    conn.execute("DELETE FROM encodings WHERE id = ?", (encoding_id,))
    conn.commit()
    conn.close()
    _encoding_cache = None


def list_encodings():
    """Return all encodings with person info for the manage UI."""
    conn = get_db()
    rows = conn.execute("""
        SELECT e.id, e.person_id, p.name as person_name, e.created_at
        FROM encodings e JOIN persons p ON e.person_id = p.id
        ORDER BY e.created_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def person_count():
    conn = get_db()
    count = conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
    conn.close()
    return count
