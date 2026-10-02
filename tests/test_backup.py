"""Backup and restore tests operate solely on temporary databases."""

import sqlite3
import sys

import pytest

from already_mentioned.database import backup
from already_mentioned.database.schema import BASE_SCHEMA


def _make_bot_database(connection: sqlite3.Connection) -> None:
    connection.executescript(";".join(BASE_SCHEMA) + ";")


def test_online_backup_verify_and_offline_restore(tmp_path) -> None:
    live_path = tmp_path / "live.db"
    copy_path = tmp_path / "copy.db"
    with sqlite3.connect(live_path) as live:
        live.execute("PRAGMA journal_mode = WAL")
        _make_bot_database(live)
        live.execute("PRAGMA user_version = 1")
        live.execute("CREATE TABLE sample (value TEXT NOT NULL)")
        live.execute("INSERT INTO sample VALUES ('original')")
        live.commit()
        assert backup.create_backup(live_path, copy_path) == 1
        live.execute("INSERT INTO sample VALUES ('later')")
        live.commit()
    assert backup.verify_backup(copy_path) == 1
    with sqlite3.connect(copy_path) as copied:
        assert copied.execute("SELECT value FROM sample").fetchall() == [("original",)]
    with pytest.raises(FileExistsError):
        backup.create_backup(live_path, copy_path)
    assert backup.restore_backup(copy_path, live_path) == 1
    with sqlite3.connect(live_path) as restored:
        assert restored.execute("SELECT value FROM sample").fetchall() == [
            ("original",)
        ]


def test_verify_rejects_broken_file_and_restore_preserves_live(tmp_path) -> None:
    broken = tmp_path / "broken.db"
    broken.write_bytes(b"not a sqlite database")
    live = tmp_path / "live.db"
    with sqlite3.connect(live) as connection:
        _make_bot_database(connection)
        connection.execute("CREATE TABLE sample (value INTEGER)")
        connection.execute("INSERT INTO sample VALUES (7)")
    with pytest.raises(sqlite3.DatabaseError):
        backup.verify_backup(broken)
    with pytest.raises(sqlite3.DatabaseError):
        backup.restore_backup(broken, live)
    with sqlite3.connect(live) as connection:
        assert connection.execute("SELECT value FROM sample").fetchone() == (7,)


def test_backup_cli_verify_and_refuses_overwrite(tmp_path, monkeypatch, capsys) -> None:
    source = tmp_path / "source.db"
    destination = tmp_path / "copy.db"
    with sqlite3.connect(source) as connection:
        _make_bot_database(connection)
    monkeypatch.setattr(
        sys, "argv", ["backup", "create", str(source), str(destination)]
    )
    backup.main()
    assert "создана и проверена" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["backup", "verify", str(destination)])
    backup.main()
    assert "Копия целостна" in capsys.readouterr().out
    monkeypatch.setattr(
        sys, "argv", ["backup", "create", str(source), str(destination)]
    )
    with pytest.raises(SystemExit) as error:
        backup.main()
    assert error.value.code == 1


def test_verify_rejects_broken_foreign_key(tmp_path) -> None:
    path = tmp_path / "orphan.db"
    with sqlite3.connect(path) as connection:
        _make_bot_database(connection)
        connection.execute("CREATE TABLE parent (id INTEGER PRIMARY KEY)")
        connection.execute(
            "CREATE TABLE child (parent_id INTEGER REFERENCES parent(id))"
        )
        connection.execute("INSERT INTO child VALUES (999)")
    with pytest.raises(ValueError, match="внешние ключи"):
        backup.verify_backup(path)
