"""SQLite connection helper."""

from pathlib import Path

import aiosqlite


async def connect_database(path: str | Path) -> aiosqlite.Connection:
    """Connect to SQLite and create the database directory if necessary."""
    database_path = Path(path)
    if str(database_path) != ":memory:":
        database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = await aiosqlite.connect(database_path)
    await connection.execute("PRAGMA foreign_keys = ON")
    await connection.execute("PRAGMA busy_timeout = 5000")
    return connection
