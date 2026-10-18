
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

# Store the database in a local folder inside the project.
DATA_DIR = Path(__file__).resolve().parent / "data"
DB_PATH = DATA_DIR / "clauseguard_history.db"


def get_connection():
    """Open a connection to the local SQLite database."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    """Create the answer history table if it does not exist."""
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS answer_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_name TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                evidence_status TEXT NOT NULL,
                evidence_quote TEXT,
                created_at TEXT NOT NULL
            )
            """
        )


def save_answer(
    document_name,
    question,
    answer,
    evidence_status,
    evidence_quote=None,
):
    """Save one answer and its evidence status locally."""
    timestamp = datetime.now(timezone.utc).isoformat()

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO answer_history (
                document_name,
                question,
                answer,
                evidence_status,
                evidence_quote,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                document_name,
                question,
                answer,
                evidence_status,
                evidence_quote,
                timestamp,
            ),
        )
        return cursor.lastrowid


def get_history(limit=100):
    """Return recent saved answers, newest first."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, document_name, question, answer,
                   evidence_status, evidence_quote, created_at
            FROM answer_history
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [dict(row) for row in rows]


def delete_answer(answer_id):
    """Delete one saved answer by its database ID."""
    with get_connection() as connection:
        connection.execute(
            "DELETE FROM answer_history WHERE id = ?",
            (answer_id,),
        )
