from dataclasses import dataclass


@dataclass
class VectorizeConfig:
    blacklevel: float = 0.5
    turdsize: int = 2
    alphamax: float = 1.0
    opttolerance: float = 0.2
