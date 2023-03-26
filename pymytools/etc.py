#!/usr/bin/env python3
"""Un-categorized feature of `pymytools`."""

from pathlib import Path
import simpleaudio as sa
from pymytools.logger import markup, console, SUPPORTING_COLORS
import time


class Me:
    """Memes about me. :D"""

    _path_to_life: Path = Path(
        Path(__file__).parent.parent, "assets/sounds/life_is.wav"
    )

    @property
    def how_is_your_life(self) -> None:

        wave_obj = sa.WaveObject.from_wave_file(self._path_to_life.as_posix())
        play_obj = wave_obj.play()
        play_obj.wait_done()
        
    @property
    def python_is(self) -> None:
        
        for c in SUPPORTING_COLORS:
            console.print("\t🔥 " + markup("STATICALLY TYPED", c, "bold")+" 🔥", end="\r")
            time.sleep(0.5)
        print("")


Sun = Me()
