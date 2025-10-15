"""Utilities for user registration and authentication."""
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, Tuple

from werkzeug.security import check_password_hash, generate_password_hash


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(BASE_DIR)
DB_PATH = os.path.join(ROOT_DIR, "users.db")


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create the user table when it does not already exist."""
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


def normalize_email(email: str) -> str:
    return email.strip().lower()


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


__all__ = [
    "DB_PATH",
    "init_db",
    "register_user",
    "authenticate_user",
]
