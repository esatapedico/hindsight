"""WP469: a prefetch aborted by shutdown's client close must not log a WARNING.

Every few seconds after a turn ended, ``logs/errors.log`` showed
``Hindsight recall failed: Server disconnected``. It was not the server:
``shutdown()`` gives up on a still-running recall after ``_join_prefetch(5.0)``,
bumps the generation, then closes the shared aiohttp session. The aborted request
surfaces in ``_do_recall`` as ``ServerDisconnectedError``. That is teardown noise,
not a server failure: it must be one INFO line with no traceback, while a genuine
disconnect outside shutdown stays a WARNING with traceback.
"""

import logging
import threading

import pytest

from conftest import FakeClient

try:
    from aiohttp.client_exceptions import ServerDisconnectedError
except ModuleNotFoundError:  # the real client always brings aiohttp; keep the suite runnable without it
    class ServerDisconnectedError(Exception):
        pass


def _loud_records(caplog):
    """WARNING-or-above records that look like the fleet's recall-failure noise."""
    return [
        record
        for record in caplog.records
        if record.levelno >= logging.WARNING
        and ("recall failed" in record.getMessage().lower() or "server disconnected" in record.getMessage().lower())
    ]


def test_shutdown_aborted_prefetch_logs_info_not_warning(provider, caplog, monkeypatch):
    """WP469: the recall shutdown abandons must be INFO (no traceback), never WARNING."""
    entered = threading.Event()
    release = threading.Event()

    def blocked_recall(query):
        entered.set()
        assert release.wait(timeout=10), "shutdown() never gave up on the prefetch"
        raise ServerDisconnectedError("Server disconnected")

    instance, _ = provider({"prefetch_waits_for_retain": False})
    monkeypatch.setattr(instance, "_recall", blocked_recall)
    instance.queue_prefetch("what is next?")
    assert entered.wait(timeout=5), "prefetch worker never started its recall"

    with caplog.at_level(logging.DEBUG):
        instance.shutdown()  # the 5.0s join expires; the client closes under the recall
        release.set()  # let the aborted request surface in _do_recall
        instance._prefetch_thread.join(10)

    assert _loud_records(caplog) == []
    cancelled = [
        record
        for record in caplog.records
        if record.levelno == logging.INFO and "prefetch cancelled at shutdown" in record.getMessage().lower()
    ]
    assert len(cancelled) == 1
    assert all(record.exc_info is None for record in cancelled)


def test_next_prefetch_after_shutdown_is_not_cancelled(provider, caplog, monkeypatch):
    """WP469: queue_prefetch() clears the flag, so a new prefetch fails loud again."""
    instance, _ = provider({"prefetch_waits_for_retain": False})
    instance._prefetch_cancel.set()  # as shutdown() leaves it

    def failing_recall(query):
        raise ServerDisconnectedError("Server disconnected")

    monkeypatch.setattr(instance, "_recall", failing_recall)
    with caplog.at_level(logging.DEBUG):
        instance.queue_prefetch("after the restart")
        instance._prefetch_thread.join(10)

    assert not instance._prefetch_cancel.is_set()
    loud = [record for record in _loud_records(caplog) if "recall failed" in record.getMessage().lower()]
    assert len(loud) == 1
    assert loud[0].exc_info is not None  # the traceback is preserved for a genuine failure
    assert not any("prefetch cancelled at shutdown" in record.getMessage().lower() for record in caplog.records)
    instance.shutdown()


def test_genuine_disconnect_outside_shutdown_stays_loud(provider, caplog, monkeypatch):
    """WP469: a ServerDisconnectedError with no teardown in progress keeps WARNING + traceback."""
    instance, _ = provider({"prefetch_waits_for_retain": False})
    assert not instance._prefetch_cancel.is_set()

    def failing_recall(query):
        raise ServerDisconnectedError("Server disconnected")

    monkeypatch.setattr(instance, "_recall", failing_recall)
    with caplog.at_level(logging.DEBUG):
        assert instance._do_recall("anything") == ("", 0)

    loud = [record for record in _loud_records(caplog) if "recall failed" in record.getMessage().lower()]
    assert len(loud) == 1
    assert "server disconnected" in loud[0].getMessage().lower()
    assert loud[0].exc_info is not None
    assert not any("prefetch cancelled at shutdown" in record.getMessage().lower() for record in caplog.records)
    instance.shutdown()


def test_retriable_connection_error_retries_once_outside_embedded(provider, monkeypatch):
    """WP469: the stale-connection retry is no longer local_embedded-only, and never fires mid-shutdown."""

    class Flaky(FakeClient):
        def __init__(self):
            super().__init__(recall_texts=["back"])
            self.attempts = 0

        async def arecall(self, **kwargs):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("Cannot connect to host 127.0.0.1:8888")
            return await super().arecall(**kwargs)

    instance, fake = provider({"prefetch_waits_for_retain": False}, client=Flaky())
    assert isinstance(fake, Flaky)
    result = instance._do_recall("q")
    assert result == ("- back", 1)
    assert fake.attempts == 2  # one retry, not a loop

    attempts = {"n": 0}

    async def dead(_client):
        attempts["n"] += 1
        raise RuntimeError("Cannot connect to host 127.0.0.1:8888")

    for flag in (instance._prefetch_cancel, instance._shutting_down):
        flag.set()
        try:
            with pytest.raises(RuntimeError):
                instance._run_hindsight_operation(dead)
        finally:
            flag.clear()
        assert attempts["n"] == 1
        attempts["n"] = 0
    instance.shutdown()


def test_join_prefetch_reports_whether_the_thread_finished(provider, monkeypatch):
    """_join_prefetch() is True once the thread is done, False while alive or missing."""
    instance, _ = provider({"prefetch_waits_for_retain": False})
    assert instance._join_prefetch(0.01) is False  # no thread yet

    entered = threading.Event()
    release = threading.Event()

    def blocked_recall(query):
        entered.set()
        assert release.wait(timeout=10)
        return []

    monkeypatch.setattr(instance, "_recall", blocked_recall)
    instance.queue_prefetch("slow")
    assert entered.wait(timeout=5)
    assert instance._join_prefetch(0.05) is False  # still running
    release.set()
    instance._prefetch_thread.join(10)
    assert instance._join_prefetch(0.05) is True  # finished
    instance.shutdown()
