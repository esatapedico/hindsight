"""Session-scoped prefetch dedup: an entry reaches the model — and Hermes' transcript, which
replays it on every later request — exactly once per session.

Hindsight re-ranks overlapping observations every turn: on a measured 94-turn session 6,308
entry slots came back but only 1,600 were distinct, so ~75% of every prompt was re-sent
duplicates that the transcript then carried forever. The plugin drops a repeat at the single
point both prefetch paths converge on (``_finish_prefetch``) and forgets them at every session
boundary, where the transcript that held them no longer exists.
"""

import pytest

import hindsight_hermes as plugin
from conftest import FakeClient

# The default header _finish_prefetch() prepends. Asserted byte for byte: an unchanged turn has
# to come back identical, because Hermes' prompt cache keys off the exact prefix bytes.
DEFAULT_HEADER = (
    "# Hindsight Memory (persistent cross-session context)\n"
    "Use this to answer questions about the user and prior sessions. "
    "Do not call tools to look up information that is already present here."
)


def _serve(fake: FakeClient, *texts: str) -> None:
    """Point the fake at the next turn's recall results (it answers every call with one list)."""
    fake._recall_texts = list(texts)


def _entries(block: str) -> list[str]:
    return [line for line in block.split("\n") if line.startswith("- ")]


def test_repeated_entry_is_injected_only_once(provider):
    instance, fake = provider(
        {"recall_sync": True},
        client=FakeClient(recall_texts=["Ada drinks espresso", "Ada ships on Fridays"]),
    )
    first = instance.prefetch("who is Ada?")
    _serve(fake, "Ada ships on Fridays", "Ada moved to Lisbon")
    second = instance.prefetch("what changed this week?")

    assert _entries(first) == ["- Ada drinks espresso", "- Ada ships on Fridays"]
    # The overlap is gone, the genuinely new entry still arrives, and nothing was rewritten.
    assert _entries(second) == ["- Ada moved to Lisbon"]
    instance.shutdown()


def test_negated_entry_is_never_mistaken_for_a_repeat(provider):
    """Exact hashes only. "deployment is feasible" and "deployment is not feasible" are ~0.88
    similar — a similarity matcher would silently drop the negation and flip the conclusion."""
    feasible, infeasible = "deployment is feasible", "deployment is not feasible"
    instance, fake = provider({"recall_sync": True}, client=FakeClient(recall_texts=[feasible]))
    assert _entries(instance.prefetch("can we deploy on Friday?")) == [f"- {feasible}"]

    # Across turns: the negation arrives alongside the already-sent claim and still gets
    # through, so "a repeat" means byte-identical, never close enough.
    _serve(fake, feasible, infeasible)
    assert _entries(instance.prefetch("are you sure?")) == [f"- {infeasible}"]
    instance.shutdown()


def test_all_repeat_prefetch_injects_nothing(provider):
    instance, fake = provider({"recall_sync": True}, client=FakeClient(recall_texts=["one", "two"]))
    instance.prefetch("first turn")
    _serve(fake, "two", "one")

    assert instance.prefetch("second turn") == ""
    # No header either: Hermes must inject nothing at all, and the indicator must not report a
    # count for memories that were not sent.
    assert instance.recall_status() is None
    instance.shutdown()


def test_prefetch_without_repeats_is_byte_identical(provider):
    texts = ["fact one", "fact two"]
    instance, fake = provider({"recall_sync": True}, client=FakeClient(recall_texts=texts))
    block = instance.prefetch("what do you know?")

    assert block == f"{DEFAULT_HEADER}\n\n" + "\n".join(f"- {t}" for t in texts)

    # A later turn of entirely different entries is byte-identical too — the no-drop path must
    # return the original string, never a re-serialized one.
    _serve(fake, "fact three")
    assert instance.prefetch("anything else?") == f"{DEFAULT_HEADER}\n\n- fact three"
    assert instance.recall_status().count == 1
    instance.shutdown()


def test_indicator_counts_only_injected_entries(provider):
    instance, fake = provider({"recall_sync": True}, client=FakeClient(recall_texts=["one", "two", "three"]))
    instance.prefetch("turn one")
    assert instance.recall_status().count == 3

    _serve(fake, "one", "two", "four", "five")
    instance.prefetch("turn two")

    # "one" and "two" were dropped; "four" and "five" are new. The indicator reports what was
    # injected, not what recall returned.
    assert instance.recall_status().count == 2
    instance.shutdown()


def test_background_prefetch_path_dedups_too(provider):
    """The default (recall_sync off) path: queue_prefetch() warms a result that the NEXT
    prefetch() drains. It converges on the same hook, so repeats must be dropped here as well."""
    instance, fake = provider({})
    _serve(fake, "Ada drinks espresso")
    instance.queue_prefetch("who is Ada?")
    first = instance.prefetch("who is Ada?")
    _serve(fake, "Ada drinks espresso")
    instance.queue_prefetch("what does Ada drink?")
    second = instance.prefetch("what does Ada drink?")

    assert _entries(first) == ["- Ada drinks espresso"]
    assert second == ""
    instance.shutdown()


def test_dropped_entry_takes_its_provenance_with_it(provider):
    """Indented lines under a bullet are that entry's source, not standalone prose — dropping the
    bullet must not orphan them into a headerless fragment."""
    instance, _ = provider({})
    drop = instance._drop_repeated_entries
    assert drop("- Ada drinks espresso\n  source: session-1")[0] == "- Ada drinks espresso\n  source: session-1"

    kept, dropped = drop("- Ada drinks espresso\n  source: session-1\n- Ada moved to Lisbon\n  source: session-2")

    # The repeat takes its source lines with it; the genuinely new entry keeps its own.
    assert kept == "- Ada moved to Lisbon\n  source: session-2"
    assert dropped == 1
    instance.shutdown()


def test_unexpected_shapes_fail_open(provider):
    """Nothing here is a repeat, so everything must come back untouched — an odd recall shape may
    never cost the turn its memory (or raise into the reply path)."""
    instance, _ = provider({})
    drop = instance._drop_repeated_entries

    assert drop("")[0] == ""  # empty turn
    assert drop("- one\n- two") == ("- one\n- two", 0)  # first sighting
    for prose in ("Synthesis with no bullets at all.", "lead-in:\n  still prose"):
        assert drop(prose) == (prose, 0)  # no entries to compare
    for shape in ("-", "*", "\n\n", "- alone", "* star", "-  spaced  "):
        assert drop(shape) == (shape, 0)  # degenerate entry shapes are still no repeat
    # CRLF and a trailing newline survive a no-op turn untouched. (New content: the entries above
    # are seen by now, and a trailing \r is not part of an entry's identity.)
    assert drop("- alpha\r\n- beta\r\n") == ("- alpha\r\n- beta\r\n", 0)
    instance.shutdown()


def test_recall_tool_output_is_not_filtered(provider):
    """Dedup is for the injected sidecar only. A model that explicitly calls hindsight_recall
    asked for these memories, so it gets them even if the same ones were injected earlier."""
    instance, _ = provider({"recall_sync": True}, client=FakeClient(recall_texts=["Ada drinks espresso"]))
    assert _entries(instance.prefetch("what does Ada drink?")) == ["- Ada drinks espresso"]

    result = instance.handle_tool_call("hindsight_recall", {"query": "what does Ada drink?"})

    assert result == '{"result": "1. Ada drinks espresso"}'
    instance.shutdown()


BOUNDARIES = [
    ("on_session_start", (), {}),
    ("on_session_start", (), {"boundary_reason": "compression"}),
    ("on_session_end", ([{"role": "user", "content": "bye"}],), {}),
    ("on_session_reset", (), {}),
    ("on_session_switch", ("session-2",), {}),
]


@pytest.mark.parametrize(
    "hook,args,kwargs",
    BOUNDARIES,
    ids=[f"{hook}-{kwargs.get('boundary_reason', 'plain')}" for hook, _args, kwargs in BOUNDARIES],
)
def test_session_boundary_forgets_seen_entries(provider, hook, args, kwargs):
    """After any boundary the transcript no longer holds what was recorded — including after a
    durable compression, which rewrites it — so the entries must be deliverable again."""
    instance, _ = provider({"recall_sync": True}, client=FakeClient(recall_texts=["fact one"]))
    instance.prefetch("who is Ada?")
    assert instance._seen_entry_dashes, "prefetch recorded nothing to forget"

    getattr(instance, hook)(*args, **kwargs)

    assert instance._seen_entry_dashes == set()
    assert _entries(instance.prefetch("who is Ada?")) == ["- fact one"]
    instance.shutdown()


def test_session_hooks_delegate_to_the_base_class(provider, monkeypatch):
    """The overrides only forget the dedup set — Hermes owns the rest of each hook, including
    retaining the transcript handed to on_session_end(). The plugin must not swallow it."""
    seen = []
    monkeypatch.setattr(
        plugin.MemoryProvider,
        "on_session_end",
        lambda self, *args, **kwargs: seen.append((args, kwargs)),
        raising=False,
    )
    transcript = [{"role": "user", "content": "bye"}]
    instance, _ = provider({})

    instance.on_session_end(transcript, reason="shutdown")

    assert seen == [((transcript,), {"reason": "shutdown"})]
    instance.shutdown()