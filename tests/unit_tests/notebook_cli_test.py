from __future__ import annotations

from importlib import import_module

notebook_main = import_module("notebook_cli.main")


def test_notebook_main_can_exit(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _: "3")

    notebook_main.main()

    output = capsys.readouterr().out
    assert "ModelForge Notebook UI" in output


def test_prompt_index_retries_until_valid(monkeypatch, capsys):
    answers = iter(["x", "4", "2"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    index = notebook_main._prompt_index("Select: ", 3)

    assert index == 1
    assert "Enter a number between 1 and 3." in capsys.readouterr().out
