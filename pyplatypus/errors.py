"""Errors that have to survive a trip into R.

The R package catches these and re-raises them as R conditions. Whatever crosses the
bridge must therefore be plain data — no exception objects, no tracebacks, nothing that
needs a Python stack to make sense of. A clinician reading the message in an R console
is the audience.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError


def _render_location(location: tuple[Any, ...]) -> str:
    """Turn pydantic's ('models', 0, 'filters') into 'models[0].filters'.

    Union members show up in the location as the matched variant's name; those entries
    are noise to a user who never asked about unions, so they are dropped.
    """
    parts: list[str] = []
    for item in location:
        if isinstance(item, int):
            if parts:
                parts[-1] = f"{parts[-1]}[{item}]"
            else:
                parts.append(f"[{item}]")
        elif isinstance(item, str) and item.endswith("]"):
            continue  # pydantic's tagged-union marker, e.g. "focal[FocalLossSpec]"
        else:
            parts.append(str(item))
    return ".".join(parts) if parts else "(top level)"


class PlatypusError(Exception):
    """Base class, so R can catch one thing."""

    kind = "platypus_error"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "message": str(self)}


class ConfigError(PlatypusError):
    """A spec that does not describe a runnable experiment.

    Carries the individual problems as data, because "three things are wrong" is far
    more useful than the first one pydantic happened to hit.
    """

    kind = "config_error"

    def __init__(self, message: str, problems: list[dict[str, str]] | None = None,
                 source: str | None = None):
        self.problems = problems or []
        self.source = source
        super().__init__(message)

    @classmethod
    def from_validation_error(cls, error: ValidationError,
                              source: str | None = None) -> ConfigError:
        problems = []
        for raw in error.errors():
            message = raw.get("msg", "invalid value")
            for noise in ("Value error, ", "Assertion failed, "):
                message = message.removeprefix(noise)
            entry = {
                "where": _render_location(raw.get("loc", ())),
                "problem": message,
            }
            if "input" in raw:
                shown = repr(raw["input"])
                entry["got"] = shown if len(shown) <= 120 else shown[:117] + "..."
            problems.append(entry)

        where = f" in {source}" if source else ""
        count = len(problems)
        noun = "problem" if count == 1 else "problems"
        header = f"The configuration{where} has {count} {noun}:"
        body = "\n".join(
            f"  - {p['where']}: {p['problem']}"
            + (f" (got {p['got']})" if "got" in p else "")
            for p in problems
        )
        return cls(f"{header}\n{body}", problems=problems, source=source)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "message": str(self),
            "source": self.source,
            "problems": self.problems,
        }
