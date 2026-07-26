import json
from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class ExpComponent:
    def title(self):
        return ''

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
        # return [f.name for f in fields(self)
        #     if not clean or getattr(self, f.name) is not None
        # ]
        return list(self.asDict(clean).keys())

    def values(self, clean=True):
        return list(self.asDict(clean).values())

    def get_values(self, vorder):
        ''' returns field values in prescribed order '''
        return [getattr(self, f) for f in vorder]

    def set_field(self, f, v, replace=True):
        ''' sets all instances of field to value in self and any nested Components '''
        for field in self.fields(clean=False):
            child = getattr(self, field)
            if field == f and (replace or getattr(self, field) is None):
                setattr(self, field, v)
            elif isinstance(child, ExpComponent):
                child.set_field(f, v)

    def copy(self, values=True, clean=False):
        self_type = type(self)
        new_set = self_type()
        if not values:
            return new_set

        for a in self.fields(clean):
            val = getattr(self, a)
            setattr(new_set, a, val.copy() if isinstance(val, ExpComponent) else val)
        return new_set


@dataclass
class CompSet(ExpComponent):
    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, item, value):
        setattr(self, item, value)

    def field_count(self, clean=True) -> int:
        return len(self.fields(clean))

    def has_field(self, key, clean=True):
        return key in self.asDict(clean)

    def has_value(self, value):
        return value in self.asDict().values()

    def set_all(self, value, clean=False):
        ''' sets all parts to value, if clean is false, then it will also overwrite None values '''
        for f in self.fields(clean):
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
