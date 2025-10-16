"""Utilities for user registration, authentication, and posture storage."""
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from werkzeug.security import check_password_hash, generate_password_hash


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(BASE_DIR)
DB_PATH = os.path.join(ROOT_DIR, "users.db")


NORMAL_LABELS: Tuple[str, ...] = ("정상", "normal", "Normal")


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create the core tables when they do not already exist."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS posture_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                posture_label TEXT NOT NULL,
                score REAL,
                recorded_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_posture_logs_user_time ON posture_logs(user_id, recorded_at)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_baselines (
                user_id INTEGER PRIMARY KEY,
                baseline TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )


def normalize_email(email: str) -> str:
    return email.strip().lower()


def _coerce_timestamp(value: Optional[object]) -> Optional[datetime]:
    if value is None:
        return datetime.utcnow()
    if isinstance(value, (int, float)):
        return datetime.utcfromtimestamp(float(value))
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return datetime.utcnow()
        try:
            dt = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
        except ValueError:
            return None
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    return None


def _coerce_score(value: Optional[object]) -> Tuple[bool, Optional[float]]:
    if value is None:
        return True, None
    try:
        return True, float(value)
    except (TypeError, ValueError):
        return False, None


def _fetch_user_id(conn: sqlite3.Connection, email: str) -> Optional[int]:
    cursor = conn.execute(
        "SELECT id FROM users WHERE email = ?",
        (email,),
    )
    row = cursor.fetchone()
    return int(row[0]) if row else None


def _normal_label_placeholders() -> str:
    return ", ".join(["?"] * len(NORMAL_LABELS))


def register_user(email: str, password: str) -> Tuple[bool, Optional[str]]:
    """Insert a new user when the email is unused."""
    if not isinstance(email, str) or not isinstance(password, str):
        return False, "email_and_password_required"

    email = normalize_email(email)
    if not email:
        return False, "email_required"
    if len(password) < 6:
        return False, "password_too_short"

    password_hash = generate_password_hash(password)
    created_at = datetime.utcnow().isoformat()

    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
                (email, password_hash, created_at),
            )
    except sqlite3.IntegrityError:
        return False, "email_already_used"

    return True, None


def authenticate_user(email: str, password: str) -> Tuple[bool, Optional[str]]:
    if not isinstance(email, str) or not isinstance(password, str):
        return False, "email_and_password_required"

    email = normalize_email(email)
    if not email:
        return False, "email_required"

    with get_connection() as conn:
        cursor = conn.execute(
            "SELECT password_hash FROM users WHERE email = ?",
            (email,),
        )
        row = cursor.fetchone()

    if row is None:
        return False, "user_not_found"

    password_hash = row[0]
    if not check_password_hash(password_hash, password):
        return False, "invalid_credentials"

    return True, None


def record_posture_event(
    email: str,
    label: str,
    *,
    score: Optional[object] = None,
    recorded_at: Optional[object] = None,
) -> Tuple[bool, Optional[str]]:
    """Persist a posture prediction for the given user."""
    if not isinstance(email, str):
        return False, "email_required"
    normalized_email = normalize_email(email)
    if not normalized_email:
        return False, "email_required"

    if not isinstance(label, str) or not label.strip():
        return False, "label_required"
    trimmed_label = label.strip()

    timestamp = _coerce_timestamp(recorded_at)
    if timestamp is None:
        return False, "invalid_timestamp"
    score_ok, score_value = _coerce_score(score)
    if not score_ok:
        return False, "invalid_score"

    with get_connection() as conn:
        user_id = _fetch_user_id(conn, normalized_email)
        if user_id is None:
            return False, "user_not_found"
        conn.execute(
            """
            INSERT INTO posture_logs (user_id, posture_label, score, recorded_at)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, trimmed_label, score_value, timestamp.isoformat()),
        )

    return True, None


def store_user_baseline(email: str, baseline: List[float]) -> Tuple[bool, Optional[str]]:
    """Persist the 21-value baseline posture for the user."""
    if not isinstance(email, str):
        return False, "email_required"
    normalized_email = normalize_email(email)
    if not normalized_email:
        return False, "email_required"

    if not isinstance(baseline, list) or len(baseline) != 21:
        return False, "baseline_invalid"

    try:
        coerced = [float(value) for value in baseline]
    except (TypeError, ValueError):
        return False, "baseline_invalid"

    timestamp = datetime.utcnow().isoformat()
    encoded = json.dumps(coerced, ensure_ascii=False)

    with get_connection() as conn:
        user_id = _fetch_user_id(conn, normalized_email)
        if user_id is None:
            return False, "user_not_found"
        conn.execute(
            """
            INSERT INTO user_baselines (user_id, baseline, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                baseline = excluded.baseline,
                updated_at = excluded.updated_at
            """,
            (user_id, encoded, timestamp, timestamp),
        )

    return True, None


def fetch_user_baseline(email: str) -> Tuple[Optional[List[float]], Optional[str]]:
    """Load the stored baseline posture for the user."""
    if not isinstance(email, str):
        return None, "email_required"
    normalized_email = normalize_email(email)
    if not normalized_email:
        return None, "email_required"

    with get_connection() as conn:
        user_id = _fetch_user_id(conn, normalized_email)
        if user_id is None:
            return None, "user_not_found"
        row = conn.execute(
            "SELECT baseline FROM user_baselines WHERE user_id = ?",
            (user_id,),
        ).fetchone()

    if row is None:
        return None, "baseline_not_set"

    try:
        values = json.loads(row[0])
        if not isinstance(values, list) or len(values) != 21:
            return None, "baseline_corrupted"
        coerced = [float(value) for value in values]
    except (TypeError, ValueError, json.JSONDecodeError):
        return None, "baseline_corrupted"

    return coerced, None


def get_posture_stats(
    email: str,
    *,
    days: int = 7,
) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
    if not isinstance(email, str):
        return None, "email_required"
    normalized_email = normalize_email(email)
    if not normalized_email:
        return None, "email_required"

    try:
        days_int = int(days)
    except (TypeError, ValueError):
        return None, "invalid_days"

    if days_int <= 0:
        return None, "invalid_days"

    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=days_int)
    start_iso = start_dt.isoformat()
    end_iso = end_dt.isoformat()

    hourly: List[Dict[str, object]] = []
    weekday: List[Dict[str, object]] = []
    labels: List[Dict[str, object]] = []

    placeholders = _normal_label_placeholders()
    hourly_query = f"""
        SELECT strftime('%H', recorded_at) AS hour,
               COUNT(*) AS total,
               SUM(CASE WHEN posture_label NOT IN ({placeholders}) THEN 1 ELSE 0 END) AS bad,
               AVG(score) AS avg_score
        FROM posture_logs
        WHERE user_id = ? AND recorded_at BETWEEN ? AND ?
        GROUP BY hour
        ORDER BY hour
    """
    weekday_query = f"""
        SELECT strftime('%w', recorded_at) AS weekday,
               COUNT(*) AS total,
               SUM(CASE WHEN posture_label NOT IN ({placeholders}) THEN 1 ELSE 0 END) AS bad
        FROM posture_logs
        WHERE user_id = ? AND recorded_at BETWEEN ? AND ?
        GROUP BY weekday
        ORDER BY weekday
    """

    with get_connection() as conn:
        user_id = _fetch_user_id(conn, normalized_email)
        if user_id is None:
            return None, "user_not_found"

        hourly_rows = conn.execute(
            hourly_query,
            (*NORMAL_LABELS, user_id, start_iso, end_iso),
        ).fetchall()
        weekday_rows = conn.execute(
            weekday_query,
            (*NORMAL_LABELS, user_id, start_iso, end_iso),
        ).fetchall()
        label_rows = conn.execute(
            """
            SELECT posture_label, COUNT(*) AS cnt
            FROM posture_logs
            WHERE user_id = ? AND recorded_at BETWEEN ? AND ?
            GROUP BY posture_label
            ORDER BY cnt DESC
            """,
            (user_id, start_iso, end_iso),
        ).fetchall()

    for hour, total, bad, avg_score in hourly_rows:
        entry: Dict[str, object] = {
            "hour": f"{hour}:00",
            "total": int(total),
            "bad": int(bad or 0),
        }
        if avg_score is not None:
            entry["avg_score"] = float(round(avg_score, 2))
        hourly.append(entry)

    weekday_names = ["일", "월", "화", "수", "목", "금", "토"]
    for weekday_idx, total, bad in weekday_rows:
        index = int(weekday_idx) % len(weekday_names)
        entry = {
            "weekday": weekday_names[index],
            "total": int(total),
            "bad": int(bad or 0),
        }
        weekday.append(entry)

    total_events = 0
    for label, count in label_rows:
        count_int = int(count)
        labels.append({"label": label, "count": count_int})
        total_events += count_int

    summary: Dict[str, object] = {
        "range": {"start": start_iso, "end": end_iso},
        "hourly": hourly,
        "weekday": weekday,
        "labels": labels,
        "total_events": total_events,
    }

    return summary, None


__all__ = [
    "DB_PATH",
    "init_db",
    "register_user",
    "authenticate_user",
    "record_posture_event",
    "get_posture_stats",
    "store_user_baseline",
    "fetch_user_baseline",
]
