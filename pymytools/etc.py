#!/usr/bin/env python3
"""Un-categorized feature of `pymytools`."""
import time

import simpleaudio as sa
from pkg_resources import resource_filename

import pymytools
from pymytools.logger import console
from pymytools.logger import markup
from pymytools.logger import SUPPORTING_COLORS


class Me:
    """Memes about me. :D"""

    _path_to_life: str = resource_filename(
        pymytools.__name__, "assets/sounds/life_is.wav"
    )

    @property
    def how_is_your_life(self) -> None:
        wave_obj = sa.WaveObject.from_wave_file(self._path_to_life)
        play_obj = wave_obj.play()
        play_obj.wait_done()

    @property
    def python_is(self) -> None:
        for c in SUPPORTING_COLORS:
            console.print(
                "\t🔥 " + markup("STATICALLY TYPED", c, "bold") + " 🔥", end="\r"
            )
            time.sleep(0.5)
        print("")


Sun = Me()
