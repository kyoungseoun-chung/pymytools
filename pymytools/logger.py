#!/usr/bin/env python3
"""Logging tools"""
import logging

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

console = Console()

from dataclasses import dataclass, field
import time

logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(markup=True)],
)

log = logging.getLogger("rich")


@dataclass
class Timer:
    """Pystops timer class. You can import this module with name `Timer`.
    This class is intended to compute elapsed time of computation with name flag.

    >>> Timer
    >>> Timer.start("sim")
    >>> time.sleep(1) # Any computation
    >>> Timer.end("sim")
    >>> Timer.elapsed("sim)
    ~ 1 # in second

    """

    clock: dict[str, dict[str, float]] = field(default_factory=dict)

    def start(self, name: str) -> None:
        """Start timer with name"""

        now = time.perf_counter()

        if name in self.clock:
            self.clock[name]["start"] = now
        else:
            self.clock[name] = {"start": now, "elapsed": 0.0}
            self.clock[name]["n_calls"] = 0

    def end(self, name: str) -> None:
        """End timer with name"""

        try:
            now = time.perf_counter()
            elapsed = now - self.clock[name]["start"]
            self.clock[name]["elapsed"] += elapsed
            self.clock[name]["n_calls"] += 1
        except KeyError:
            raise RuntimeError("Timer: timer is not started!")

    def elapsed(self, name: str) -> float:
        """Return elapsed time with name"""

        return self.clock[name]["elapsed"]

    def display(self) -> None:
        table = Table(title="Timer Report")
        table.add_column("Name", justify="center", style="cyan")
        table.add_column(r"Elapsed time \[s]", justify="center", style="green")
        table.add_column("#Calls", justify="center", style="magenta")

        for k, v in self.clock.items():
            table.add_row(k, f"{v['elapsed']:.5f}", f"{v['n_calls']}")

        console.print(table)


timer = Timer()


def markup(msg: str, color: str | None = None, style: str | None = None) -> str:
    """Generate rich markup string.

    Args:
        color (str): Color of the text. Color should be one of basic terminal 16 colors, ["black", "red", "green", "yellow", "blue", "magenta", "cyan", "white", "bright_black", "bright_red", "bright_green", "bright_yellow", "bright_blue", "bright_magenta", "bright_cyan", "bright_white"]
        msg (str): Message to be printed.
        style (str): Style of the text. See rich style list. One of "bold", "italic", "underline", "strike".
    """

    tag = "["

    if style is None:
        tag += ""
    else:
        assert style in ["bold", "italic", "underline", "strike"] or style in [
            "b",
            "i",
            "u",
            "s",
        ]
        tag += style

    if color is None:
        tag += ""
    else:
        assert color in [
            "black",
            "red",
            "green",
            "yellow",
            "blue",
            "magenta",
            "cyan",
            "white",
            "bright_black",
            "bright_red",
            "bright_green",
            "bright_yellow",
            "bright_blue",
            "bright_magenta",
            "bright_cyan",
            "bright_white",
        ]

        if len(tag) > 1:
            tag += " " + color
        else:
            tag += color

    tag += "]"

    return f"{tag}{msg}[/{tag[1:]}"


def rich_report(num_calls: int, elapsed_time: float, name: str) -> Table:
    """Report timer using rich table."""

    table = Table(title="Timer Report")
    table.add_column("Name", justify="center", style="cyan")
    table.add_column("Number of calls [-]", justify="center", style="magenta")
    table.add_column(r"Elapsed time \[s]", justify="center", style="green")

    table.add_row(name, f"{num_calls}", f"{elapsed_time:.5f}")

    return table


def draw_rule(width: int = 80, character: str = "=", color: str = "green") -> str:
    """Draw a rule for a given width and color."""

    if character == "-":
        character = chr(9472)
    elif character == "=":
        character = chr(9552)
    else:
        msg = f"Invalid line style ({character})! Should be one of (-) or (=)."
        raise ValueError(msg)

    rule = ""

    for _ in range(width):
        rule += character

    console.print(markup(rule, color=color, style="bold"))

    return rule
