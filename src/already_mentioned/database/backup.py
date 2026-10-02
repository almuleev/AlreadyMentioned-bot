"""SQLite online backup, verification, and deliberate offline restore CLI."""

import argparse
import sqlite3
from pathlib import Path

from already_mentioned.database.schema import REQUIRED_COLUMNS, SCHEMA_VERSION


def _open_readonly(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise ValueError(f"Файл SQLite не найден: {path}")
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)


def verify_backup(path: Path) -> int:
    """Check page integrity and foreign keys; return the schema version."""
    with _open_readonly(path) as connection:
        result = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if result != "ok":
            raise ValueError("Проверка целостности SQLite не пройдена")
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise ValueError("В копии нарушены внешние ключи")
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise ValueError("Версия схемы копии новее версии этого бота")
        for table, required in REQUIRED_COLUMNS.items():
            columns = {
                row[1] for row in connection.execute(f"PRAGMA table_info({table})")
            }
            if not required <= columns:
                raise ValueError(f"В копии нет обязательной структуры таблицы {table}")
        return version


def create_backup(source: Path, destination: Path) -> int:
    """Use SQLite's online backup API instead of copying a live database file."""
    if source.resolve() == destination.resolve():
        raise ValueError("Источник и копия должны быть разными файлами")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _open_readonly(source) as live:
        with destination.open("xb"):
            pass
        try:
            with sqlite3.connect(destination) as backup:
                live.backup(backup)
            return verify_backup(destination)
        except BaseException:
            destination.unlink(missing_ok=True)
            raise


def restore_backup(source: Path, destination: Path) -> int:
    """Restore a verified backup with SQLite backup API; caller must stop the bot."""
    if source.resolve() == destination.resolve():
        raise ValueError("Источник и рабочая база должны быть разными файлами")
    version = verify_backup(source)
    if not destination.is_file():
        raise ValueError("Рабочая база не найдена; проверьте путь восстановления")
    with _open_readonly(source) as backup:
        with sqlite3.connect(destination) as live:
            backup.backup(live)
    verify_backup(destination)
    return version


def main() -> None:
    parser = argparse.ArgumentParser(description="Резервное копирование SQLite бота")
    subcommands = parser.add_subparsers(dest="operation", required=True)
    create = subcommands.add_parser("create", help="Создать копию работающей базы")
    create.add_argument("source", type=Path)
    create.add_argument("destination", type=Path)
    verify = subcommands.add_parser("verify", help="Проверить копию")
    verify.add_argument("path", type=Path)
    restore = subcommands.add_parser(
        "restore", help="Восстановить проверенную копию при остановленном боте"
    )
    restore.add_argument("source", type=Path)
    restore.add_argument("destination", type=Path)
    args = parser.parse_args()
    try:
        if args.operation == "create":
            version = create_backup(args.source, args.destination)
            print(
                f"Копия создана и проверена: {args.destination}; версия схемы {version}"
            )
        elif args.operation == "verify":
            version = verify_backup(args.path)
            print(f"Копия целостна: {args.path}; версия схемы {version}")
        else:
            version = restore_backup(args.source, args.destination)
            print(f"База восстановлена: {args.destination}; версия схемы {version}")
    except (OSError, sqlite3.Error, ValueError) as exc:
        parser.exit(1, f"Ошибка SQLite: {exc}\n")


if __name__ == "__main__":
    main()
