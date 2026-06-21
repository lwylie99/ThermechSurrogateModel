import json
from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class Component:
    def asDict(self, clean=True):
        if not clean:
            return asdict(self)

        non_none = lambda d: {
            k: non_none(v) for k, v in d.items() if v is not None
        } if isinstance(d, dict) else d
        return non_none(asdict(self))

    def asJson(self, clean=True):
        return json.dumps(self.asDict(clean), sort_keys=False, indent=4)

    def fields(self, clean=True):
        return list(self.asDict(clean).keys())

    def values(self, clean=True):
        return list(self.asDict(clean).values())

    def title(self):
        return ''


@dataclass
class CompSet(Component):
    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, item, value):
        setattr(self, item, value)

    def field_count(self) -> int:
        return len(self.fields())

    def set(self, value):
        for f in self.fields(clean=False):
            setattr(self, f, value)
        return self


@dataclass
class EdgeSet(CompSet):
    top: Any = None
    bottom: Any = None
    left: Any = None
    right: Any = None


@dataclass
class PinnSet(EdgeSet):
    core: Any = None


@dataclass
class ModularComponent(Component):
    x: float = None
    y: float = None

    def title(self):
        return f'Component at ({self.x}, {self.y})'


@dataclass
class NDComponent(ModularComponent):
    measure: str = None
    length: float | int = None
    width: float | int = None
    masks = PinnSet(
        top=(0, slice(None)),  # y=0, all x   → (1, 20) = 20 pts
        bottom=(-1, slice(None)),  # y=max, all x → 20 pts
        left=(slice(None), 0),  # all y, x=0   → 10 pts
        right=(slice(None), -1),  # all y, x=max → 10 pts
        core=(slice(1, -1), slice(1, -1))  # (8, 18) = 144 pts
    )
    # 0 for x-normal (left/right), 1 for y-normal (top/bottom)
    axis = EdgeSet(top=1, bottom=1, left=0, right=0)
    # direction of normal vector pointing outward
    out = EdgeSet(top=1, bottom=-1, left=-1, right=1)

    def shape(self) -> tuple:
        return self.width, self.length  # (ny, nx) = (rows, cols) — numpy convention
