from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IDSResult:
    char: str
    structure: str
    components: list[str]


def parse_ids(char: str) -> IDSResult:
    """Parse IDS structure for a character.

    Placeholder: returns empty structure until IDS sources are integrated.
    """
    return IDSResult(char=char, structure="", components=[])
