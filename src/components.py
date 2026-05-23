import json
from dataclasses import dataclass, asdict, fields
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

    def asStr(self, clean=True):
        return json.dumps(self.asDict(clean), sort_keys=False, indent=4)

    def keys(self, clean=True):
        return list(self.asDict(clean).keys())

    def values(self, clean=True):
        return list(self.asDict(clean).values())

    def get(self, feild_name):
        return getattr(self, feild_name)


@dataclass
class ModularComponent(Component):
    x: float = None
    y: float = None


@dataclass
class NDComponent(ModularComponent):
    measure: str = None
    length: float | int = None
    width: float | int = None

    def dims(self) -> tuple:
        return self.length, self.width


@dataclass
class PartSet(Component):
    top: Any = None
    bottom: Any = None
    left: Any = None
    right: Any = None
    core: Any = None

    def parts(self):
        return [f.name for f in fields(self)]
