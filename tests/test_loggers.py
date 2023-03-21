#!/usr/bin/env python3
"""Test loggers"""
import time

import pytest

from pymytools.logger import draw_rule
from pymytools.logger import markup
from pymytools.logger import Report
from pymytools.logger import timer


def test_rich_table() -> None:
    content = {
        "name of test": ["test1", "test2"],
        "value": [1, 2],
        "anything else": [0.1, 0.2],
    }

    report = Report(
        "test", content, ["left", "center", "right"], ["green", "red", "blue"]
    )

    report.display()


def test_rich_markup() -> None:
    assert (
        markup("test", color="yellow", style="bold")
        == "[bold yellow]test[/bold yellow]"
    )


def test_timer_class() -> None:
    timer.start("test")
    for _ in range(5):
        time.sleep(0.1)

    timer.end("test")

    assert timer.elapsed("test") == pytest.approx(0.5, 0.5)

    # Secondary
    timer.start("test")
    for _ in range(5):
        time.sleep(0.1)

    timer.end("test")

    assert timer.elapsed("test") == pytest.approx(1, 0.1)

    # With other name
    timer.start("other")
    for _ in range(5):
        time.sleep(0.1)

    timer.end("other")

    assert timer.elapsed("other") == pytest.approx(0.5, 0.1)

    timer.display()


def test_ruler() -> None:
    ruler = draw_rule()

    assert ruler == chr(9552) * 80
