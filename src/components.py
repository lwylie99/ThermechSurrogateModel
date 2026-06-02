import json
import os
from dataclasses import dataclass, asdict, fields
from typing import Any

from torch import Tensor


@dataclass
class Component:
    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, item, value):
        setattr(self, item, value)

    def asDict(self, clean=True):
        if not clean:
            return asdict(self)

        non_none = lambda d: {
            k: non_none(v) for k, v in d.items() if v is not None
        } if isinstance(d, dict) else d
        return non_none(asdict(self))

    def asJson(self, clean=True):
        return json.dumps(self.asDict(clean), sort_keys=False, indent=4)

    def keys(self, clean=True):
        return list(self.asDict(clean).keys())

    def values(self, clean=True):
        return list(self.asDict(clean).values())


@dataclass
class EdgeSet(Component):
    top: Any = None
    bottom: Any = None
    left: Any = None
    right: Any = None

@dataclass
class PartSet(EdgeSet):
    core: Any = None

@dataclass
class ModularComponent(Component):
    x: float = None
    y: float = None

@dataclass
class NDComponent(ModularComponent):
    measure: str = None
    length: float | int = None
    width: float | int = None
    masks = PartSet(
        top=(0, slice(None)),
        bottom=(-1, slice(None)),
        left=(slice(None), 0),
        right=(slice(None), -1),
        core=(slice(1, -1), slice(1, -1))
    )
    # 0 for x-normal (left/right), 1 for y-normal (top/bottom)
    axis = EdgeSet(top=1, bottom=1, left=0, right=0)
    # direction of normal vector pointing outward
    out = EdgeSet(top=1, bottom=-1, left=-1, right=1)

    # def __post_init__(self):

    def shape(self) -> tuple:
        return self.length, self.width
