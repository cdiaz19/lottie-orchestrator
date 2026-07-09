from pathlib import Path

from lottie.memory.base import NullMemoryClient
from lottie.memory.mock import MockMemoryClient
from lottie.memory.store import SqliteMemoryClient, build_memory_client


def test_factory_sqlite(tmp_path: Path) -> None:
    client = build_memory_client(tmp_path, backend="sqlite", path=".lottie/memory.db")
    assert isinstance(client, SqliteMemoryClient)


def test_factory_mock(tmp_path: Path) -> None:
    assert isinstance(build_memory_client(tmp_path, backend="mock", path="x"), MockMemoryClient)


def test_factory_null_default(tmp_path: Path) -> None:
    assert isinstance(build_memory_client(tmp_path, backend="null", path="x"), NullMemoryClient)
