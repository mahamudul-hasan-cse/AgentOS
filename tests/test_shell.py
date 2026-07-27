"""Tests for the shell's command-resolution layer only — no network calls.

`resolve()` must map each command string to the right handler with the right
parsed args, and turn malformed input into clean errors (never exceptions)."""

import pytest

from shell.repl import (
    COMMANDS,
    handle_cat,
    handle_find,
    handle_kill,
    handle_mem,
    handle_ps,
    handle_run,
    handle_strace,
    resolve,
)


def test_every_command_resolves_to_its_handler():
    # a representative mapping of command name -> handler function
    expected = {
        "ps": handle_ps,
        "kill": handle_kill,
        "cat": handle_cat,
        "find": handle_find,
        "mem": handle_mem,
        "strace": handle_strace,
        "run": handle_run,
    }
    for name, handler in expected.items():
        res = resolve(name if name in ("ps",) else f"{name} x")
        assert res.command is not None
        assert res.command.name == name
        assert res.command.handler is handler


def test_no_arg_commands_parse_clean():
    for name in ("ps", "top", "help", "exit", "quit"):
        res = resolve(name)
        assert res.ok
        assert res.command.name == name
        assert res.args == []


def test_single_token_args_are_parsed():
    assert resolve("kill P1").args == ["P1"]
    assert resolve("mem agent-7").args == ["agent-7"]
    assert resolve("strace 5").args == ["5"]


def test_optional_arg_commands():
    # default (no arg)
    assert resolve("limits").ok and resolve("limits").args == []
    assert resolve("ls").ok and resolve("ls").args == []
    # explicit arg
    assert resolve("limits alice").args == ["alice"]
    assert resolve("ls bob").args == ["bob"]


def test_rest_of_line_commands_keep_the_whole_remainder():
    # 'run' and 'find' take the entire remainder as one argument
    r = resolve("run write a haiku about paging")
    assert r.ok and r.command.name == "run"
    assert r.args == ["write a haiku about paging"]

    f = resolve("find how do plants make energy")
    assert f.ok and f.args == ["how do plants make energy"]

    # 'cat' takes the rest of the line too (filenames may contain spaces)
    assert resolve("cat my notes.txt").args == ["my notes.txt"]


def test_missing_required_args_produce_clear_errors_not_exceptions():
    for line in ("kill", "mem", "cat", "find", "run"):
        res = resolve(line)
        assert not res.ok
        assert res.error is not None
        assert "usage" in res.error.lower()


def test_too_many_args_are_rejected():
    assert not resolve("kill P1 P2 P3").ok  # kill accepts at most "-t <pid>"
    assert not resolve("mem a b").ok
    assert not resolve("ps extra").ok  # ps takes none
    assert "no arguments" in resolve("ps extra").error


def test_kill_accepts_the_tree_flag():
    res = resolve("kill -t P1")
    assert res.ok
    assert res.command.name == "kill"
    assert res.args == ["-t", "P1"]


def test_kill_rejects_two_positionals_at_the_handler():
    """`kill P1 P2` is arity-legal (the parser allows two tokens so "-t <pid>"
    fits) but meaningless, so the handler rejects it — before any network call."""
    from shell.repl import Context, ShellError, handle_kill

    res = resolve("kill P1 P2")
    assert res.ok
    ctx = Context(base_url="http://unused.invalid", agent="tester")
    with pytest.raises(ShellError):
        handle_kill(ctx, res.args)


def test_unknown_command_is_a_clean_error():
    res = resolve("frobnicate xyz")
    assert not res.ok
    assert res.command is None
    assert "unknown command" in res.error.lower()


def test_blank_input_is_a_benign_noop():
    for line in ("", "   ", "\t"):
        res = resolve(line)
        assert res.empty
        assert not res.ok
        assert res.error is None


def test_leading_and_trailing_whitespace_tolerated():
    res = resolve("   kill   P9   ")
    assert res.ok
    assert res.command.name == "kill"
    assert res.args == ["P9"]


def test_all_registered_commands_have_usage_and_help():
    for name, cmd in COMMANDS.items():
        assert cmd.usage and cmd.help
        assert cmd.arg_style in ("none", "tokens", "rest")
