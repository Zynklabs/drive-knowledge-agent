"""Sync state in SQLite: LangChain's ``RecordManager`` contract plus two small tables.

The record manager remembers which chunk hashes are in the vector store and which file
(group) each belongs to, so re-syncs skip unchanged chunks and clean up stale ones. It also
remembers the Drive version each file was indexed from, so re-syncs skip downloading files
that have not changed.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.indexing import RecordManager

STATE_DB = "state.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS upsertion_record (
    namespace  TEXT NOT NULL,
    key        TEXT NOT NULL,
    group_id   TEXT,
    updated_at REAL NOT NULL,
    PRIMARY KEY (namespace, key)
);
CREATE INDEX IF NOT EXISTS ix_record_group ON upsertion_record (namespace, group_id);
CREATE INDEX IF NOT EXISTS ix_record_time ON upsertion_record (namespace, updated_at);
CREATE TABLE IF NOT EXISTS meta (
    namespace TEXT NOT NULL,
    name      TEXT NOT NULL,
    value     TEXT NOT NULL,
    PRIMARY KEY (namespace, name)
);
CREATE TABLE IF NOT EXISTS file_version (
    namespace TEXT NOT NULL,
    file_id   TEXT NOT NULL,
    version   TEXT NOT NULL,
    PRIMARY KEY (namespace, file_id)
);
"""


class SQLiteRecordManager(RecordManager):
    def __init__(self, namespace: str, db_path: Path) -> None:
        super().__init__(namespace)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(db_path)
        self.create_schema()

    def close(self) -> None:
        self._db.close()

    # --- RecordManager -------------------------------------------------------

    def create_schema(self) -> None:
        with self._db:
            self._db.executescript(_SCHEMA)

    def get_time(self) -> float:
        return time.time()

    def update(
        self,
        keys: Sequence[str],
        *,
        group_ids: Sequence[str | None] | None = None,
        time_at_least: float | None = None,
    ) -> None:
        now = self.get_time()
        if time_at_least is not None and now < time_at_least:
            raise ValueError(f"clock went backwards: {now} < {time_at_least}")
        groups = list(group_ids) if group_ids is not None else [None] * len(keys)
        if len(groups) != len(keys):
            raise ValueError("keys and group_ids must have the same length")
        with self._db:
            self._db.executemany(
                "INSERT INTO upsertion_record (namespace, key, group_id, updated_at) "
                "VALUES (?, ?, ?, ?) ON CONFLICT (namespace, key) "
                "DO UPDATE SET group_id = excluded.group_id, updated_at = excluded.updated_at",
                [(self.namespace, k, g, now) for k, g in zip(keys, groups, strict=True)],
            )

    def exists(self, keys: Sequence[str]) -> list[bool]:
        found: set[str] = set()
        for batch in _batches(keys):
            marks = ",".join("?" * len(batch))
            rows = self._db.execute(
                f"SELECT key FROM upsertion_record WHERE namespace = ? AND key IN ({marks})",  # noqa: S608
                [self.namespace, *batch],
            )
            found.update(row[0] for row in rows)
        return [key in found for key in keys]

    def list_keys(
        self,
        *,
        before: float | None = None,
        after: float | None = None,
        group_ids: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[str]:
        sql = "SELECT key FROM upsertion_record WHERE namespace = ?"
        params: list[Any] = [self.namespace]
        if before is not None:
            sql += " AND updated_at < ?"
            params.append(before)
        if after is not None:
            sql += " AND updated_at > ?"
            params.append(after)
        if group_ids is not None:
            sql += f" AND group_id IN ({','.join('?' * len(group_ids))})"
            params.extend(group_ids)
        sql += " ORDER BY key"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return [row[0] for row in self._db.execute(sql, params)]

    def delete_keys(self, keys: Sequence[str]) -> None:
        with self._db:
            for batch in _batches(keys):
                marks = ",".join("?" * len(batch))
                self._db.execute(
                    f"DELETE FROM upsertion_record WHERE namespace = ? AND key IN ({marks})",  # noqa: S608
                    [self.namespace, *batch],
                )

    # The indexing API is used synchronously; async variants delegate.
    async def acreate_schema(self) -> None:
        self.create_schema()

    async def aget_time(self) -> float:
        return self.get_time()

    async def aupdate(
        self,
        keys: Sequence[str],
        *,
        group_ids: Sequence[str | None] | None = None,
        time_at_least: float | None = None,
    ) -> None:
        self.update(keys, group_ids=group_ids, time_at_least=time_at_least)

    async def aexists(self, keys: Sequence[str]) -> list[bool]:
        return self.exists(keys)

    async def alist_keys(
        self,
        *,
        before: float | None = None,
        after: float | None = None,
        group_ids: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[str]:
        return self.list_keys(before=before, after=after, group_ids=group_ids, limit=limit)

    async def adelete_keys(self, keys: Sequence[str]) -> None:
        self.delete_keys(keys)

    # --- extras ------------------------------------------------------------------

    def list_groups(self) -> list[str]:
        rows = self._db.execute(
            "SELECT DISTINCT group_id FROM upsertion_record "
            "WHERE namespace = ? AND group_id IS NOT NULL ORDER BY group_id",
            [self.namespace],
        )
        return [row[0] for row in rows]

    def count(self) -> int:
        row = self._db.execute(
            "SELECT COUNT(*) FROM upsertion_record WHERE namespace = ?", [self.namespace]
        ).fetchone()
        return int(row[0])

    def clear(self) -> None:
        """Forget every record and file version in this namespace (meta is kept)."""
        with self._db:
            self._db.execute("DELETE FROM upsertion_record WHERE namespace = ?", [self.namespace])
            self._db.execute("DELETE FROM file_version WHERE namespace = ?", [self.namespace])

    def file_versions(self) -> dict[str, str]:
        """Version each file was indexed from, for files that still have records."""
        rows = self._db.execute(
            "SELECT file_id, version FROM file_version v WHERE namespace = ? AND EXISTS ("
            "SELECT 1 FROM upsertion_record r "
            "WHERE r.namespace = v.namespace AND r.group_id = v.file_id)",
            [self.namespace],
        )
        return {row[0]: row[1] for row in rows}

    def set_file_version(self, file_id: str, version: str) -> None:
        with self._db:
            self._db.execute(
                "INSERT INTO file_version (namespace, file_id, version) VALUES (?, ?, ?) "
                "ON CONFLICT (namespace, file_id) DO UPDATE SET version = excluded.version",
                [self.namespace, file_id, version],
            )

    def delete_file_version(self, file_id: str) -> None:
        with self._db:
            self._db.execute(
                "DELETE FROM file_version WHERE namespace = ? AND file_id = ?",
                [self.namespace, file_id],
            )

    def get_meta(self, name: str) -> Any:
        row = self._db.execute(
            "SELECT value FROM meta WHERE namespace = ? AND name = ?", [self.namespace, name]
        ).fetchone()
        return json.loads(row[0]) if row else None

    def set_meta(self, name: str, value: Any) -> None:
        with self._db:
            self._db.execute(
                "INSERT INTO meta (namespace, name, value) VALUES (?, ?, ?) "
                "ON CONFLICT (namespace, name) DO UPDATE SET value = excluded.value",
                [self.namespace, name, json.dumps(value)],
            )


def _batches(keys: Sequence[str], size: int = 500) -> list[Sequence[str]]:
    return [keys[i : i + size] for i in range(0, len(keys), size)]
