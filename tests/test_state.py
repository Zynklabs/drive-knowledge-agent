import asyncio
from pathlib import Path

import pytest

from drive_agent.indexing.state import SQLiteRecordManager


@pytest.fixture
def records(tmp_path: Path) -> SQLiteRecordManager:
    rm = SQLiteRecordManager("ns", tmp_path / "state.db")
    yield rm  # type: ignore[misc]
    rm.close()


def test_update_exists_and_groups(records: SQLiteRecordManager) -> None:
    records.update(["a", "b", "c"], group_ids=["F1", "F1", "F2"])
    assert records.exists(["a", "c", "z"]) == [True, True, False]
    assert records.list_groups() == ["F1", "F2"]
    assert records.list_keys(group_ids=["F1"]) == ["a", "b"]
    assert records.count() == 3


def test_upsert_moves_group_and_time(records: SQLiteRecordManager) -> None:
    records.update(["a"], group_ids=["F1"])
    first = records.list_keys(after=0)
    t = records.get_time()
    records.update(["a"], group_ids=["F2"], time_at_least=t)
    assert first == ["a"]
    assert records.list_keys(group_ids=["F2"]) == ["a"]
    assert records.list_keys(before=t) == []  # refreshed timestamp
    assert records.list_keys(after=t - 1, limit=1) == ["a"]


def test_time_at_least_in_future_is_rejected(records: SQLiteRecordManager) -> None:
    with pytest.raises(ValueError, match="clock went backwards"):
        records.update(["a"], time_at_least=records.get_time() + 3600)


def test_mismatched_group_ids(records: SQLiteRecordManager) -> None:
    with pytest.raises(ValueError, match="same length"):
        records.update(["a", "b"], group_ids=["F1"])


def test_delete_and_clear(records: SQLiteRecordManager) -> None:
    records.update([f"k{i}" for i in range(1200)], group_ids=["G"] * 1200)  # > one SQL batch
    records.delete_keys([f"k{i}" for i in range(600)])
    assert records.count() == 600
    assert all(records.exists([f"k{i}" for i in range(600, 1200)]))
    records.clear()
    assert records.count() == 0


def test_namespaces_are_isolated(tmp_path: Path) -> None:
    one = SQLiteRecordManager("one", tmp_path / "s.db")
    two = SQLiteRecordManager("two", tmp_path / "s.db")
    one.update(["a"], group_ids=["F"])
    one.set_meta("embedding", "local:x")
    assert two.exists(["a"]) == [False]
    assert two.get_meta("embedding") is None
    assert one.get_meta("embedding") == "local:x"
    one.close()
    two.close()


def test_meta_roundtrip_and_persistence(tmp_path: Path) -> None:
    rm = SQLiteRecordManager("ns", tmp_path / "s.db")
    rm.set_meta("last_sync", {"files": 2, "at": "now"})
    rm.set_meta("last_sync", {"files": 3, "at": "later"})
    rm.close()
    again = SQLiteRecordManager("ns", tmp_path / "s.db")
    assert again.get_meta("last_sync") == {"files": 3, "at": "later"}
    again.close()


def test_async_variants_delegate(records: SQLiteRecordManager) -> None:
    async def scenario() -> list[str]:
        await records.acreate_schema()
        await records.aupdate(
            ["a", "b"], group_ids=["F", "F"], time_at_least=await records.aget_time()
        )
        assert await records.aexists(["a"]) == [True]
        await records.adelete_keys(["b"])
        return await records.alist_keys(group_ids=["F"])

    assert asyncio.run(scenario()) == ["a"]
