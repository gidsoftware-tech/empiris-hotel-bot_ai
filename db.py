"""SQLite storage layer for the Grand Hotel reservation assistant.

One shared connection guarded by a lock, so it is safe to use from Flask's
worker threads. Creating a booking is a single atomic transaction that
re-checks inventory, so two guests can never be given the same last room.

Nothing here knows about conversation state; bot.py owns that.
"""
import os
import sqlite3
import threading
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

# room type -> sleeps (max guests), rate per night (KES), rooms in inventory
ROOMS = {
    "standard": {"sleeps": 2, "rate": 12000, "total": 10},
    "deluxe":   {"sleeps": 3, "rate": 18000, "total": 6},
    "family":   {"sleeps": 5, "rate": 25000, "total": 4},
    "suite":    {"sleeps": 2, "rate": 35000, "total": 2},
}
MAX_NIGHTS = 30

DB_PATH = os.environ.get("DB_PATH") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "reservations.db")

LOCK = threading.RLock()

_db = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)
_db.row_factory = sqlite3.Row
_db.execute("""CREATE TABLE IF NOT EXISTS bookings(
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    name      TEXT    NOT NULL,
    room      TEXT    NOT NULL,
    guests    INTEGER NOT NULL,
    check_in  TEXT    NOT NULL,   -- ISO date, e.g. 2026-10-20
    check_out TEXT    NOT NULL,   -- ISO date, exclusive (the check-out morning)
    nights    INTEGER NOT NULL,
    rate      INTEGER NOT NULL,
    total     INTEGER NOT NULL)""")
_db.execute("CREATE INDEX IF NOT EXISTS idx_bookings_room "
            "ON bookings(room, check_in, check_out)")

# ---------- Users (login) ----------
_db.execute("""CREATE TABLE IF NOT EXISTS users(
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    username   TEXT    UNIQUE NOT NULL,
    password_hash TEXT   NOT NULL,
    role       TEXT    NOT NULL DEFAULT 'guest'
)""")
_db.commit()

# seed default users if none exist (override with env vars)
_with_users = _db.execute("SELECT COUNT(*) FROM users").fetchone()[0]
if _with_users == 0:
    _db.execute("INSERT INTO users(username,password_hash,role) VALUES(?,?,?)",
                (os.environ.get("ADMIN_USER", "admin"),
                 generate_password_hash(os.environ.get("ADMIN_PASSWORD", "admin123")),
                 "admin"))
    _db.execute("INSERT INTO users(username,password_hash,role) VALUES(?,?,?)",
                (os.environ.get("GUEST_USER", "guest"),
                 generate_password_hash(os.environ.get("GUEST_PASSWORD", "guest123")),
                 "guest"))
    _db.commit()
_db.execute("""CREATE TABLE IF NOT EXISTS pos_items(
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    name      TEXT    NOT NULL,
    category  TEXT    NOT NULL,
    price_kes INTEGER NOT NULL)""")
_db.execute("""CREATE TABLE IF NOT EXISTS pos_charges(
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    guest_name  TEXT    NOT NULL,
    item_name   TEXT    NOT NULL,
    price_kes   INTEGER NOT NULL,
    qty         INTEGER NOT NULL DEFAULT 1,
    total_kes   INTEGER NOT NULL,
    created_at  TEXT    NOT NULL)""")
_db.execute("CREATE INDEX IF NOT EXISTS idx_pos_charges_guest "
            "ON pos_charges(guest_name, created_at)")

_POS_SEED = [
    ("Tusker Beer", "drinks", 500),
    ("White Cap",   "drinks", 500),
    ("Soda",        "drinks", 150),
    ("Water",       "drinks", 100),
    ("Nyama Choma", "food", 1500),
    ("Ugali & Sukuma", "food", 400),
    ("Samosa",      "food", 200),
    ("Chapati",     "food", 50),
    ("Massage (60 min)", "spa", 3000),
    ("Sauna",       "spa", 1000),
    ("Airport Shuttle", "transport", 2000),
    ("Late Checkout",   "services", 2000),
]
for _name, _cat, _price in _POS_SEED:
    _exists = _db.execute("SELECT 1 FROM pos_items WHERE name=? AND category=?",
                          (_name, _cat)).fetchone()
    if not _exists:
        _db.execute("INSERT INTO pos_items(name,category,price_kes) VALUES(?,?,?)",
                    (_name, _cat, _price))

_db.execute("""CREATE TABLE IF NOT EXISTS pos_payments(
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    guest_name  TEXT    NOT NULL,
    amount_kes  INTEGER NOT NULL,
    method      TEXT    NOT NULL,
    receipt_no  TEXT    NOT NULL,
    created_at  TEXT    NOT NULL)""")
_db.execute("CREATE INDEX IF NOT EXISTS idx_pos_payments_guest "
            "ON pos_payments(guest_name, created_at)")
_db.commit()

_COLS = "id,name,room,guests,check_in,check_out,nights,rate,total"

# match a full name, or just the first name of a longer name
_NAME_SQL = "(lower(name)=lower(?) OR lower(name) LIKE lower(?) || ' %' ESCAPE '\\')"


def _like_escape(s):
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def rooms_free(conn, room, check_in, check_out):
    """How many rooms of `room` type are unbooked for [check_in, check_out)."""
    booked = conn.execute(
        "SELECT COUNT(*) FROM bookings WHERE room=? AND check_in<? AND check_out>?",
        (room, check_out.isoformat(), check_in.isoformat())).fetchone()[0]
    return max(ROOMS[room]["total"] - booked, 0)


def book_room(name, room, guests, check_in, check_out):
    """Book one room of `room` type for the given date range.

    Returns the created booking as a dict, or None when nothing is free.
    """
    if room not in ROOMS:
        return None
    nights = (check_out - check_in).days
    with LOCK:
        try:
            _db.execute("BEGIN IMMEDIATE")          # write lock for check + insert
            if rooms_free(_db, room, check_in, check_out) <= 0:
                _db.rollback()
                return None
            rate = ROOMS[room]["rate"]
            cur = _db.execute(
                "INSERT INTO bookings(name,room,guests,check_in,check_out,nights,rate,total) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (name, room, guests, check_in.isoformat(), check_out.isoformat(),
                 nights, rate, rate * nights))
            _db.commit()
            return get_booking(cur.lastrowid)
        except Exception:
            _db.rollback()
            raise


def get_booking(booking_id):
    with LOCK:
        row = _db.execute(f"SELECT {_COLS} FROM bookings WHERE id=?",
                          (booking_id,)).fetchone()
    return dict(row) if row else None


def find_bookings(name, since_iso):
    """Bookings whose name matches (full name or first name) and are not over."""
    with LOCK:
        rows = _db.execute(
            f"SELECT {_COLS} FROM bookings WHERE {_NAME_SQL} AND check_out>=? "
            "ORDER BY check_in",
            (name, _like_escape(name), since_iso)).fetchall()
    return [dict(r) for r in rows]


def list_upcoming(since_iso, limit=200):
    with LOCK:
        rows = _db.execute(
            f"SELECT {_COLS} FROM bookings WHERE check_out>=? ORDER BY check_in LIMIT ?",
            (since_iso, limit)).fetchall()
    return [dict(r) for r in rows]


def delete_bookings(ids):
    with LOCK:
        _db.executemany("DELETE FROM bookings WHERE id=?", [(i,) for i in ids])
        _db.commit()


# ---------------- POS functions ----------------
def pos_items():
    with LOCK:
        rows = _db.execute("SELECT id,name,category,price_kes FROM pos_items ORDER BY category,name").fetchall()
    return [dict(r) for r in rows]


def add_charge(guest_name, item_name, price_kes, qty=1):
    """Add a POS charge to a guest's folio. Returns the charge dict."""
    total = price_kes * qty
    created = datetime.now().date().isoformat()
    with LOCK:
        try:
            _db.execute("BEGIN IMMEDIATE")
            _db.execute(
                "INSERT INTO pos_charges(guest_name,item_name,price_kes,qty,total_kes,created_at) "
                "VALUES(?,?,?,?,?,?)",
                (guest_name, item_name, price_kes, qty, total, created))
            _db.commit()
            new_id = _db.execute("SELECT last_insert_rowid()").fetchone()[0]
            row = _db.execute("SELECT id,guest_name,item_name,price_kes,qty,total_kes,created_at "
                              "FROM pos_charges WHERE id=?", (new_id,)).fetchone()
            return dict(row) if row else None
        except Exception:
            _db.rollback()
            raise


def get_folio(guest_name, since_iso):
    """All charges for a guest on/after since_iso."""
    with LOCK:
        rows = _db.execute(
            "SELECT id,guest_name,item_name,price_kes,qty,total_kes,created_at "
            "FROM pos_charges WHERE lower(guest_name)=lower(?) AND created_at>=? "
            "ORDER BY created_at",
            (guest_name, since_iso)).fetchall()
    return [dict(r) for r in rows]


def folio_total(guest_name, since_iso):
    with LOCK:
        row = _db.execute(
            "SELECT COALESCE(SUM(total_kes),0) FROM pos_charges "
            "WHERE lower(guest_name)=lower(?) AND created_at>=?",
            (guest_name, since_iso)).fetchone()
    return int(row[0]) if row else 0


def payments_total(guest_name, since_iso):
    with LOCK:
        row = _db.execute(
            "SELECT COALESCE(SUM(amount_kes),0) FROM pos_payments "
            "WHERE lower(guest_name)=lower(?) AND created_at>=?",
            (guest_name, since_iso)).fetchone()
    return int(row[0]) if row else 0


def outstanding(guest_name, since_iso):
    return folio_total(guest_name, since_iso) - payments_total(guest_name, since_iso)


def record_payment(guest_name, amount_kes, method):
    """Record a payment and generate a receipt number. Returns the payment dict."""
    import uuid as _uuid
    receipt_no = "RCPT-" + datetime.now().strftime("%Y%m%d") + "-" + _uuid.uuid4().hex[:6].upper()
    created = datetime.now().date().isoformat()
    with LOCK:
        try:
            _db.execute("BEGIN IMMEDIATE")
            _db.execute(
                "INSERT INTO pos_payments(guest_name,amount_kes,method,receipt_no,created_at) "
                "VALUES(?,?,?,?,?)",
                (guest_name, amount_kes, method, receipt_no, created))
            _db.commit()
            new_id = _db.execute("SELECT last_insert_rowid()").fetchone()[0]
            row = _db.execute("SELECT id,guest_name,amount_kes,method,receipt_no,created_at "
                              "FROM pos_payments WHERE id=?", (new_id,)).fetchone()
            return dict(row) if row else None
        except Exception:
            _db.rollback()
            raise


def folio_details(guest_name, since_iso):
    """Charges + payments for a guest, used to build a receipt/folio view."""
    with LOCK:
        charges = _db.execute(
            "SELECT id,item_name,price_kes,qty,total_kes,created_at FROM pos_charges "
            "WHERE lower(guest_name)=lower(?) AND created_at>=? ORDER BY created_at",
            (guest_name, since_iso)).fetchall()
        payments = _db.execute(
            "SELECT id,amount_kes,method,receipt_no,created_at FROM pos_payments "
            "WHERE lower(guest_name)=lower(?) AND created_at>=? ORDER BY created_at",
            (guest_name, since_iso)).fetchall()
    return {
        "charges": [dict(r) for r in charges],
        "payments": [dict(r) for r in payments],
        "folio_total": folio_total(guest_name, since_iso),
        "paid_total": payments_total(guest_name, since_iso),
        "outstanding": outstanding(guest_name, since_iso),
    }


def stats_summary(since_iso):
    """Aggregated stats for the dashboard."""
    with LOCK:
        bookings = _db.execute("SELECT COUNT(*) FROM bookings WHERE check_out>=?", (since_iso,)).fetchone()[0] or 0
        charges = _db.execute("SELECT COALESCE(SUM(total_kes),0) FROM pos_charges WHERE created_at>=?", (since_iso,)).fetchone()[0] or 0
        payments = _db.execute("SELECT COALESCE(SUM(amount_kes),0) FROM pos_payments WHERE created_at>=?", (since_iso,)).fetchone()[0] or 0
    return {
        "total_bookings": int(bookings),
        "todays_charges": int(charges),
        "paid_today": int(payments),
        "outstanding": int(charges) - int(payments),
    }


def distinct_guests_with_charges(since_iso):
    """Guest names that have charges on/after since_iso."""
    with LOCK:
        rows = _db.execute(
            "SELECT DISTINCT guest_name FROM pos_charges WHERE created_at>=? ORDER BY guest_name",
            (since_iso,)).fetchall()
    return [r[0] for r in rows]


# ---------- Users (login) ----------
def find_user(username):
    with LOCK:
        row = _db.execute("SELECT id,username,password_hash,role FROM users WHERE username=?",
                          (username,)).fetchone()
    return dict(row) if row else None


def verify_user(username, password):
    u = find_user(username)
    if not u:
        return None
    return u if check_password_hash(u["password_hash"], password) else None
