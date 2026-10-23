"""OpenJTalk readings via pyopenjtalk-plus (the front end piper-plus uses).

`ojt` uses the default g2p settings, which is what piper-plus 2.0.2 runs at synthesis
time (incl. pyopenjtalk-plus' Sudachi homograph fix). `ojt_vanilla` is plain OpenJTalk.
No custom dictionary is used.
"""

import pyopenjtalk

MODES = {"ojt": {}, "ojt_vanilla": {"use_vanilla": True}}


def read(text: str, mode: str = "ojt") -> str:
    if mode not in MODES:
        raise ValueError(f"unknown OpenJTalk mode {mode!r}; expected one of {sorted(MODES)}")
    return pyopenjtalk.g2p(text, kana=True, **MODES[mode])
